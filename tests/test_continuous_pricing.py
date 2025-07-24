import pytest

from passengersim import Simulation, demo_network
from passengersim.config import Config
from passengersim.summaries import SimulationTables

DEFAULT_TOLERANCE = dict(rtol=1e-04, atol=1e-06)


@pytest.mark.parametrize(
    "cp_algorithm,cp_record_highest_closed_as_open",
    [
        ("None", False),
        ("BP", False),
        ("BP", True),
    ],
)
def test_3mkt_continuous_pricing(dataframe_regression, cp_algorithm, cp_record_highest_closed_as_open):
    input_file = demo_network("3MKT/11-probp-daily")
    cfg = Config.from_yaml(input_file)
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 100
    cfg.simulation_controls.burn_samples = 40
    cfg.outputs.reports.clear()
    cfg.outputs.reports.add("*")

    # turn on continuous pricing
    cfg.carriers.AL1.cp_algorithm = cp_algorithm
    cfg.carriers.AL2.cp_algorithm = cp_algorithm
    cfg.carriers.AL1.cp_bounds = 0.5
    cfg.carriers.AL2.cp_bounds = 0.5

    cfg.carriers.AL1.cp_record_highest_closed_as_open = cp_record_highest_closed_as_open
    cfg.carriers.AL2.cp_record_highest_closed_as_open = cp_record_highest_closed_as_open

    sim = Simulation(cfg)
    summary = sim.run(summarizer=SimulationTables)
    assert isinstance(summary, SimulationTables)
    dataframe_regression.check(summary.carriers)
