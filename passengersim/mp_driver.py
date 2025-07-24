import json
import pathlib

import joblib

from .config import Config
from .core import SimulationEngine
from .driver import BaseSimulation, Simulation
from .summaries import GenericSimulationTables, SimulationTables
from .summary import SummaryTables
from .utils.caffeine import keep_awake
from .utils.tempdir import MaybeTemporaryDirectory


def _subprocess_run_trial(
    trial_id: int,
    cfg_json: str,
    output_dir: pathlib.Path | None = None,
    *,
    summarizer=SimulationTables,
):
    cfg = Config.model_validate(json.loads(cfg_json))
    if cfg.db is not None and cfg.db.filename is not None and str(cfg.db.filename) != ":memory:":
        cfg.db.filename = cfg.db.filename.with_suffix(f".trial{trial_id:02}" + cfg.db.filename.suffix)
    output_dir = MaybeTemporaryDirectory(output_dir)
    sim = Simulation(cfg, output_dir.joinpath(f"passengersim-trial-{trial_id:02}"))
    summary = sim.run(single_trial=trial_id, summarizer=summarizer)
    # Passing a database connection between processes is not allowed,
    # so we need to delete it before returning the summary. But first
    # we will run any queries that were requested as output reports
    if isinstance(summary, GenericSimulationTables):
        summary.run_queries(items=cfg.outputs.reports)
    try:
        summary.cnx = None
    except AttributeError as err:
        print(err)
    return summary


class MultiSimulation(BaseSimulation):
    def __init__(
        self,
        config: Config,
        output_dir: pathlib.Path | None = None,
    ):
        super().__init__(config, output_dir)
        self.config = config
        self._simulators = {}
        # self._placeholder_sim = None

    def run(self, *, summarizer=SimulationTables):
        if self.config.raw_license_certificate is None:
            try:
                from passengersim_license import raw_license_certificate
            except ImportError:
                raw_license_certificate = None
            self.config.raw_license_certificate = raw_license_certificate
        with keep_awake():
            with joblib.Parallel(n_jobs=self.config.simulation_controls.num_trials) as parallel:
                cfg_json = self.config.model_dump_json()
                results = parallel(
                    joblib.delayed(_subprocess_run_trial)(
                        trial_id,
                        cfg_json,
                        self.output_dir,
                        summarizer=summarizer,
                    )
                    for trial_id in range(self.config.simulation_controls.num_trials)
                )
            result = summarizer.aggregate(results)
            result.config = self.config.model_copy(deep=True)
        return result

    def sequential_run(self, *, summarizer=SimulationTables):
        results = []
        cfg_json = self.config.model_dump_json()
        for trial_id in range(self.config.simulation_controls.num_trials):
            print("starting trial", trial_id)
            results.append(_subprocess_run_trial(trial_id, cfg_json, self.output_dir, summarizer=summarizer))
            print("finished trial", trial_id)
        return SummaryTables.aggregate(results)

    @property
    def _sim(self) -> SimulationEngine:
        for s in self._simulators.values():
            if isinstance(s, SimulationEngine):
                return s
        raise TypeError("No SimulationEngine found in MultiSimulation")
