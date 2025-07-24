import json
import multiprocessing
import pathlib
import time
import warnings
from contextlib import nullcontext
from datetime import UTC, datetime

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from .callbacks import CallbackMixin
from .config import Config
from .core import SimulationEngine
from .driver import BaseSimulation, Simulation
from .summaries import GenericSimulationTables, SimulationTables
from .summary import SummaryTables
from .utils import multiproc
from .utils.caffeine import keep_awake
from .utils.tempdir import MaybeTemporaryDirectory


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
    callbacks=None,
):
    cfg = Config.model_validate(json.loads(cfg_json))
    if cfg.db is not None and cfg.db.filename is not None and str(cfg.db.filename) != ":memory:":
        cfg.db.filename = cfg.db.filename.with_suffix(f".trial{trial_id:02}" + cfg.db.filename.suffix)
    output_dir = MaybeTemporaryDirectory(output_dir)

    sim = Simulation(cfg, output_dir.joinpath(f"passengersim-trial-{trial_id:02}"))
    if callbacks is not None:
        for cb in callbacks.get("begin_sample_callbacks", []):
            sim.begin_sample_callback(cb)
        for cb in callbacks.get("end_sample_callbacks", []):
            sim.end_sample_callback(cb)
        for cb in callbacks.get("daily_callbacks", []):
            sim.daily_callback(cb)
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


class MultiSimulation(BaseSimulation, CallbackMixin):
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
                    trial_id,
                    cfg_json,
                    output_dir=self.output_dir,
                    summarizer=summarizer,
                    callbacks=self.callback_functions(),
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
        Dump the configuration to a JSON string suitable for a subprocess.

        License info is added to the config if available.  Output file
        reporting is removed.

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
        return self.config.model_dump_json(exclude={"outputs": {"html", "pickle", "excel", "log_reports"}})

    def run(
        self,
        *,
        summarizer=SimulationTables,
        output_dir: pathlib.Path | None = None,
        max_processes: int | None = None,
        rich_progress: Progress | None = None,
    ):
        """
        Run the simulation using multiple processes.

        Parameters
        ----------
        summarizer : SimulationTables
        output_dir : pathlib.Path, optional
            The directory to write output files to.  If not provided, a temporary
            directory will be created.
        max_processes : int, optional
            Maximum number of processes to run simultaneously.  If not provided, the
            number of processes will be equal to the number of CPUs on the system.
        rich_progress : Progress, optional
            A rich Progress object to use for displaying progress.  If not provided,
            a new Progress object will be created.

        Returns
        -------
        SimulationTables or subclass
        """
        run_start = time.time()
        run_start_str = datetime.fromtimestamp(run_start, UTC).isoformat()

        progress_queue = multiprocessing.Queue()
        processes = []
        n_processes_started = 0
        if max_processes is None:
            max_processes = multiprocessing.cpu_count()

        task_ids = list(range(self.config.simulation_controls.num_trials))
        num_samples = self.config.simulation_controls.num_samples
        cfg_json = self._dump_config()

        results = {}

        if rich_progress is None:
            rich_progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                auto_refresh=False,
            )
            progress_context = rich_progress
        else:
            progress_context = nullcontext(rich_progress)

        with keep_awake():
            with progress_context as progress:
                task_progress_ids = {
                    task_id: progress.add_task(f"[green]Trial {task_id}", total=num_samples) for task_id in task_ids
                }

                try:
                    for task_id in task_ids:
                        p = multiproc.Process(
                            target=_subprocess_run_trial,
                            args=(task_id, cfg_json, progress_queue, output_dir),
                            kwargs={
                                "summarizer": summarizer,
                                "callbacks": self.callback_functions(),
                            },
                        )
                        processes.append(p)

                    while n_processes_started < max_processes and n_processes_started < len(processes):
                        processes[n_processes_started].start()
                        n_processes_started += 1

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
                                visible=False,
                            )
                            results[task_id] = payload
                            if n_processes_started < len(processes):
                                processes[n_processes_started].start()
                                n_processes_started += 1
                        elif message == "update":
                            progress.update(
                                task_progress_ids[task_id],
                                advance=payload,
                                refresh=True,
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

                except KeyboardInterrupt:
                    # TODO: send a signal to running processes to terminate
                    #       gracefully, and collect partial results (if any)
                    # terminate all processes so they are not orphaned
                    for p in processes:
                        try:
                            p.terminate()
                        except AttributeError:
                            pass
                    raise

        result = summarizer.aggregate([value for key, value in sorted(results.items())])
        result.config = self.config.model_copy(deep=True)
        result._metadata["time.started"] = run_start_str
        run_finished = time.time()
        result._metadata["time.runtime"] = run_finished - run_start
        result._metadata["time.finished"] = datetime.fromtimestamp(run_finished, UTC).isoformat()

        # write output files if designated
        if self.config.outputs.html and (
            self.config.outputs.disk is True or self.config.outputs.html.filename == self.config.outputs.disk
        ):
            # this will ensure the html and disk files have the same timestamp
            try:
                filenames = result.save(self.config.outputs.html.filename)
                result._metadata["outputs.html_filename"] = filenames[".html"]
                result._metadata["outputs.disk_filename"] = filenames[".pxsim"]
            except Exception as err:
                warnings.warn(f"Error writing HTML or disk file: {err}", stacklevel=2)
        else:
            if self.config.outputs.html:
                try:
                    result.to_html(self.config.outputs.html.filename, make_dirs=True)
                except Exception as err:
                    warnings.warn(f"Error writing HTML file: {err}", stacklevel=2)
            if isinstance(self.config.outputs.disk, str | pathlib.Path):
                try:
                    result.to_file(self.config.outputs.disk, make_dirs=True)
                except Exception as err:
                    warnings.warn(f"Error writing HTML file: {err}", stacklevel=2)
        if self.config.outputs.pickle:
            try:
                result.to_pickle(self.config.outputs.pickle, make_dirs="git")
            except Exception as err:
                warnings.warn(f"Error writing pickle file: {err}", stacklevel=2)

        return result
