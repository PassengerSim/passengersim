import json
import multiprocessing
import os
import pathlib
import time

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from .config import Config
from .core import SimulationEngine
from .driver import BaseSimulation, Simulation
from .summaries import GenericSimulationTables, SimulationTables
from .summary import SummaryTables
from .utils.caffeine import keep_awake


class ThrottledUpdater:
    def __init__(self, progress_queue, trial_id, throttle: float = 0.5):
        self.progress_queue = progress_queue
        self.trial_id = trial_id
        self.throttle = throttle
        self.last_time = 0
        self.n = 0

    def __call__(self, *args, **kwargs):
        self.n += 1
        if time.time() - self.last_time > self.throttle:
            self.flush()

    def flush(self):
        self.progress_queue.put((self.trial_id, "update", self.n))
        self.last_time = time.time()
        self.n = 0


def _subprocess_run_trial(
    trial_id: int,
    cfg_json: str,
    progress_queue: multiprocessing.Queue,
    output_dir: pathlib.Path | None = None,
    *,
    summarizer=SimulationTables,
):
    cfg = Config.model_validate(json.loads(cfg_json))
    if (
        cfg.db is not None
        and cfg.db.filename is not None
        and str(cfg.db.filename) != ":memory:"
    ):
        cfg.db.filename = cfg.db.filename.with_suffix(
            f".trial{trial_id:02}" + cfg.db.filename.suffix
        )
    if output_dir is None:
        import tempfile

        _tempdir = tempfile.TemporaryDirectory()
        output_dir = os.path.join(_tempdir.name, f"passengersim-trial-{trial_id:02}")

    sim = Simulation(cfg, output_dir)
    sim.sample_done_callback = ThrottledUpdater(progress_queue, trial_id, 0.5)
    summary = sim.run(single_trial=trial_id, summarizer=summarizer)
    sim.sample_done_callback.flush()
    progress_queue.put((trial_id, "finalizing", None))
    # Passing a database connection between processes is not allowed,
    # so we need to delete it before returning the summary. But first
    # we will run any queries that were requested as output reports
    if isinstance(summary, GenericSimulationTables):
        summary.run_queries(items=cfg.outputs.reports)
    try:
        summary.cnx = None
    except AttributeError as err:
        print(err)
    progress_queue.put((trial_id, "done", summary))
    return summary


class MultiSimulation(BaseSimulation):
    def __init__(
        self,
        config: Config,
        output_dir: pathlib.Path | None = None,
    ):
        super().__init__(config, output_dir)
        self.config = config
        self._placeholder_sim = None

    def sequential_run(self, *, summarizer=SimulationTables):
        results = []
        cfg_json = self.config.model_dump_json()
        for trial_id in range(self.config.simulation_controls.num_trials):
            print("starting trial", trial_id)
            results.append(
                _subprocess_run_trial(
                    trial_id, cfg_json, self.output_dir, summarizer=summarizer
                )
            )
            print("finished trial", trial_id)
        return SummaryTables.aggregate(results)

    @property
    def _sim(self) -> SimulationEngine:
        if self._placeholder_sim is None:
            self._placeholder_sim = Simulation(self.config, self.output_dir)
        return self._placeholder_sim

    def _dump_config(self) -> str:
        """
        Dump the configuration to a JSON string.

        Returns
        -------
        str
        """
        if self.config.raw_license_certificate is None:
            try:
                from passengersim_license import raw_license_certificate
            except ImportError:
                raw_license_certificate = None
            self.config.raw_license_certificate = raw_license_certificate
        return self.config.model_dump_json()

    def run(
        self, *, summarizer=SimulationTables, output_dir: pathlib.Path | None = None
    ):
        progress_queue = multiprocessing.Queue()
        processes = []

        task_ids = list(range(self.config.simulation_controls.num_trials))
        num_samples = self.config.simulation_controls.num_samples
        cfg_json = self._dump_config()

        results = {}

        with keep_awake():
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                auto_refresh=False,
            ) as progress:
                task_progress_ids = {
                    task_id: progress.add_task(
                        f"[green]Trial {task_id}", total=num_samples
                    )
                    for task_id in task_ids
                }

                for task_id in task_ids:
                    p = multiprocessing.Process(
                        target=_subprocess_run_trial,
                        args=(task_id, cfg_json, progress_queue, output_dir),
                        kwargs={"summarizer": summarizer},
                    )
                    processes.append(p)
                    p.start()

                while any(p.is_alive() for p in processes):
                    try:
                        task_id, message, payload = progress_queue.get(timeout=0.25)
                    except multiprocessing.queues.Empty:
                        continue
                    if message == "done":
                        progress.update(
                            task_progress_ids[task_id],
                            completed=num_samples,
                            description=f"Finished Trial {task_id}",
                            refresh=True,
                        )
                        results[task_id] = payload
                    elif message == "update":
                        progress.update(
                            task_progress_ids[task_id], advance=payload, refresh=True
                        )
                    elif message == "finalizing":
                        progress.update(
                            task_progress_ids[task_id],
                            description=f"Finalizing Trial {task_id}",
                            refresh=True,
                        )
                    else:
                        print(f"Unknown message {message}")

                for p in processes:
                    p.join()

        result = summarizer.aggregate([value for key, value in sorted(results.items())])
        result.config = self.config.model_copy(deep=True)
        return result
