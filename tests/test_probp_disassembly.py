from typing import Literal

import altair
import pytest
import pandas as pd

from passengersim import MultiSimulation, Simulation, demo_network
from passengersim.config import Config
from passengersim.summaries import SimulationTables

DEFAULT_TOLERANCE = dict(rtol=1e-04, atol=1e-06)


@pytest.fixture(scope="module")
def config() -> Config:
    input_file = demo_network("3MKT/11-probp-daily")
    cfg = Config.from_yaml(input_file)
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 150
    cfg.simulation_controls.burn_samples = 75
    cfg.outputs.reports.clear()
    cfg.outputs.reports.add("*")
    return cfg


def test_probp_stepwise(config: Config, dataframe_regression):
    self = Simulation(config)

    self.setup_scenario()

    # TODO test that setup is good

    # self._run_sim_single_trial(0)

    self.sim.update_db_write_flags()
    n_samples_done = 0
    trial = 0

    summarizer = SimulationTables()

    self.sim.trial = trial
    self.sim.reset_trial_counters()

    # set the number of trials completed to 1, to allow the summarizer to run
    # from inside the loop
    self.sim.num_trials_completed = 1

    for trial in range(self.sim.config.simulation_controls.num_trials):
        for sample in range(self.sim.num_samples):
            self.sim.sample = sample
            assert self.sim.config.simulation_controls.random_seed is not None
            if self.sim.config.simulation_controls.random_seed is not None:
                # check the reseeding is stable
                self.reseed(
                    [
                        self.sim.config.simulation_controls.random_seed,
                        trial,
                        sample,
                    ]
                )
                rand_check = []
                for zz in range(10):
                    rand_check.append(self.sim.random_generator.get_uniform())
                dataframe_regression.check(
                    pd.DataFrame(rand_check, columns=['rando']),
                    basename=f"rand_check_trial{trial}_sample{sample}",
                )
                # reseed again
                self.reseed(
                    [
                        self.sim.config.simulation_controls.random_seed,
                        trial,
                        sample,
                    ]
                )

            self.sim.reset_counters()
            if self.sim.sample == 0:
                self.sim.reset_trial_counters()
            self.generate_demands()

            while True:
                event = self.sim.go()
                self.run_carrier_models(event)
                if event is None or str(event) == "Done" or (event[0] == "Done"):
                    assert (
                        self.sim.num_events() == 0
                    ), f"Event queue still has {self.sim.num_events()} events"
                    break

            n_samples_done += 1
            self.end_sample()

            partial_summary = summarizer.extract(self)

            partial_tables = [
                "bid_price_history",
                "displacement_history",
                "legs",
                "legbuckets",
                "paths",
                "carriers",
                "demand_to_come_summary",
                "fare_class_mix",
                "segmentation_by_timeframe",
                "pathclasses",
            ]

            for table_name in partial_tables:
                df = getattr(partial_summary, table_name)
                if df is not None and len(df):
                    try:
                        dataframe_regression.check(
                            df, basename=f"trial{trial}_sample{sample}_{table_name}", default_tolerance=DEFAULT_TOLERANCE
                        )
                    except AssertionError as err:
                        print(f"Failed for {table_name} at [trial={trial}, sample={sample}]")
                        raise AssertionError(f"Failed for {table_name} at [trial={trial}, sample={sample}]") from err

            assert isinstance(partial_summary, SimulationTables)

        self.sim.num_trials_completed += 1
        self.end_trial()

