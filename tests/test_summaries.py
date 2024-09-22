import pytest

from passengersim import Simulation, demo_network
from passengersim.config import Config
from passengersim.summaries import SimulationTables

DEFAULT_TOLERANCE = dict(rtol=2e-02, atol=1e-06)


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
    sim.run()
    return SimulationTables.extract(sim)


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


def test_table_segmentation_by_timeframe(summary, dataframe_regression):
    assert isinstance(summary, SimulationTables)
    dataframe_regression.check(
        summary.segmentation_by_timeframe.stack(0),
        basename="segmentation_by_timeframe",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_table_carriers(summary, dataframe_regression):
    assert isinstance(summary, SimulationTables)
    df = summary.carriers
    # integer tests are flakey, change to float
    i = df.select_dtypes(include=["number"]).columns
    df[i] = df[i].astype("float") * 1.00000001
    dataframe_regression.check(
        df,
        basename="carriers",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_table_legs(summary, dataframe_regression):
    assert isinstance(summary, SimulationTables)
    df = summary.legs
    # integer tests are flakey, change to float
    i = df.select_dtypes(include=["number"]).columns
    df[i] = df[i].astype("float") * 1.00000001
    dataframe_regression.check(
        df,
        basename="legs",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_table_demand_to_come(summary, dataframe_regression):
    assert isinstance(summary, SimulationTables)
    dataframe_regression.check(
        summary.demand_to_come,
        basename="demand_to_come",
        default_tolerance=DEFAULT_TOLERANCE,
    )
