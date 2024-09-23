import json
import os
import pathlib

import joblib

from .config import Config
from .core import SimulationEngine
from .driver import BaseSimulation, Simulation
from .summaries import SimulationTables
from .summary import SummaryTables


def _subprocess_run_trial(
    trial_id: int,
    cfg_json: str,
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
    summary = sim.run(single_trial=trial_id, summarizer=summarizer)
    try:
        del summary.cnx
    except AttributeError:
        pass
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

    # def run_trial(self, trial_id: int):
    #     cfg = self.config.model_copy(deep=True)
    #     if str(cfg.db.filename) != ":memory:":
    #         cfg.db.filename = cfg.db.filename.with_suffix(
    #             f".trial{trial_id:02}" + cfg.db.filename.suffix
    #         )
    #     sim = Simulation(cfg, self.output_dir)
    #     self._simulators[trial_id] = sim
    #     summary = sim.run(single_trial=trial_id)
    #     try:
    #         del summary.cnx
    #     except AttributeError:
    #         pass
    #     return summary

    def run(self, *, summarizer=SimulationTables):
        if self.config.raw_license_certificate is None:
            try:
                from passengersim_license import raw_license_certificate
            except ImportError:
                raw_license_certificate = None
            self.config.raw_license_certificate = raw_license_certificate
        with joblib.Parallel(
            n_jobs=self.config.simulation_controls.num_trials
        ) as parallel:
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
            results.append(
                _subprocess_run_trial(
                    trial_id, cfg_json, self.output_dir, summarizer=summarizer
                )
            )
            print("finished trial", trial_id)
        return SummaryTables.aggregate(results)

    @property
    def _sim(self) -> SimulationEngine:
        for s in self._simulators.values():
            if isinstance(s, SimulationEngine):
                return s
        raise TypeError("No SimulationEngine found in MultiSimulation")


# def spin(n):
#     c = cfg.model_copy(deep=True)
#     c.simulation_controls.random_seed = 42 + n
#     c.simulation_controls.num_trials = 1
#     c.simulation_controls.num_samples = 10
#     c.simulation_controls.burn_samples = 5
#     c.db.filename = c.db.filename.with_suffix(f".trial{n:02}" + c.db.filename.suffix)
#     c.simulation_controls.show_progress_bar = False
#     sim = pax.Simulation(c)
#     summary = sim.run()
#     del summary.cnx # cannot pickle DB connection
#     return summary
