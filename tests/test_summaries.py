import pytest

from passengersim import Simulation, demo_network
from passengersim.config import Config
from passengersim.summaries import SimulationTables

DEFAULT_TOLERANCE = dict(rtol=1e-04, atol=1e-06)


@pytest.fixture(scope="module")
def config() -> Config:
    input_file = demo_network("3MKT/08-untrunc-em")
    cfg = Config.from_yaml(input_file)
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 150
    cfg.simulation_controls.burn_samples = 100
    cfg.outputs.reports.clear()
    cfg.db = None
    return cfg


@pytest.fixture(scope="module")
def summary(config: Config) -> SimulationTables:
    sim = Simulation(config)
    return sim.run(summarizer=SimulationTables)


@pytest.fixture(scope="module")
def summary2(config: Config) -> SimulationTables:
    sim0 = Simulation(config)
    sim0.run_trial(0)
    sim1 = Simulation(config)
    sim1.run_trial(1)
    return SimulationTables.aggregate(
        [SimulationTables.extract(sim0), SimulationTables.extract(sim1)]
    )


def test_table_basic(summary: SimulationTables):
    assert isinstance(summary, SimulationTables)
    assert isinstance(summary.sim, Simulation)
    assert isinstance(summary.config, Config.as_reloaded)

    # for each leg, check that the sum of gt_sold for all paths equals the leg's gt_sold
    for leg in summary.sim.sim.legs:
        local_sold, total_sold = 0, 0
        for pth in summary.sim.sim.paths:
            if pth.get_leg_id(0) == leg.leg_id:
                if pth.num_legs() == 1:
                    local_sold += pth.gt_sold
                total_sold += pth.gt_sold
            if pth.num_legs() == 2:
                if pth.get_leg_id(1) == leg.leg_id:
                    total_sold += pth.gt_sold
        assert leg.gt_sold == total_sold
        assert leg.gt_sold_local == local_sold
        assert leg.avg_local() == (local_sold / total_sold) * 100

    # check that buckets have static fares attached
    assert [b.fcst_revenue for b in summary.sim.legs[101].buckets] == [
        400.0,
        300.0,
        200.0,
        150.0,
        125.0,
        100.0,
    ]


TABLES = [
    "demand_to_come",
    "fare_class_mix",
    "legs",
    "legbuckets",
    "paths",
    "carriers",
    "segmentation_by_timeframe",
    "pathclasses",
    "bid_price_history",
    "displacement_history",
]


@pytest.mark.parametrize("table_name", TABLES)
def test_table_presence_single_process(summary, dataframe_regression, table_name: str):
    assert isinstance(summary, SimulationTables)
    df = getattr(summary, table_name)
    dataframe_regression.check(
        df, basename=table_name, default_tolerance=DEFAULT_TOLERANCE
    )


@pytest.mark.parametrize("table_name", TABLES)
def test_table_presence_multi_process(summary2, dataframe_regression, table_name: str):
    assert isinstance(summary2, SimulationTables)
    df = getattr(summary2, table_name)
    dataframe_regression.check(
        df, basename=table_name, default_tolerance=DEFAULT_TOLERANCE
    )
