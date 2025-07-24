from typing import Literal

import altair
import pytest

from passengersim import MultiSimulation, Simulation, demo_network
from passengersim.config import Config
from passengersim.summaries import SimulationTables

DEFAULT_TOLERANCE = dict(rtol=1e-04, atol=1e-06)


@pytest.fixture(scope="module")
def config() -> Config:
    input_file = demo_network("3MKT/12-udp")
    cfg = Config.from_yaml(input_file)
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 150
    cfg.simulation_controls.burn_samples = 75
    cfg.outputs.reports.clear()
    cfg.outputs.reports.add("*")
    return cfg


@pytest.fixture(scope="module")
def summary(config: Config) -> SimulationTables:
    sim = Simulation(config)
    return sim.run(summarizer=SimulationTables)


@pytest.fixture(scope="module")
def summary_mp(config: Config) -> SimulationTables:
    sim = MultiSimulation(config)
    return sim.run(summarizer=SimulationTables)


TABLES = [
    "demand_to_come",
    "demand_to_come_summary",
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
def test_3mkt_dp_table_single_process(summary, dataframe_regression, table_name: str):
    assert isinstance(summary, SimulationTables)
    df = getattr(summary, table_name)
    dataframe_regression.check(df, basename=table_name, default_tolerance=DEFAULT_TOLERANCE)


@pytest.mark.parametrize("table_name", TABLES)
def test_3mkt_dp_table_multi_process(summary_mp, dataframe_regression, table_name: str):
    assert isinstance(summary_mp, SimulationTables)
    df = getattr(summary_mp, table_name)
    dataframe_regression.check(df, basename=table_name, default_tolerance=DEFAULT_TOLERANCE)


#
# def test_3mkt_dp_bookings_by_timeframe(summary, dataframe_regression):
#     assert isinstance(summary, SummaryTables)
#     dataframe_regression.check(
#         summary.bookings_by_timeframe,
#         basename="bookings_by_timeframe",
#         default_tolerance=DEFAULT_TOLERANCE,
#     )
#
#
# @pytest.mark.filterwarnings("error")
# def test_3mkt_dp_carriers(summary, dataframe_regression):
#     assert isinstance(summary, SummaryTables)
#     df = summary.carriers
#     # integer tests are flakey, change to float
#     i = df.select_dtypes(include=["number"]).columns
#     df[i] = df[i].astype("float") * 1.00000001
#     dataframe_regression.check(
#         df,
#         basename="carriers",
#         default_tolerance=DEFAULT_TOLERANCE,
#     )
#
#
# @pytest.mark.filterwarnings("error")
# def test_3mkt_dp_fare_class_mix(summary, dataframe_regression):
#     assert isinstance(summary, SummaryTables)
#     dataframe_regression.check(
#         summary.fare_class_mix,
#         basename="fare_class_mix",
#         default_tolerance=DEFAULT_TOLERANCE,
#     )
#
#
# @pytest.mark.filterwarnings("error")
# def test_3mkt_dp_raw_fare_class_mix(summary, dataframe_regression):
#     assert isinstance(summary, SummaryTables)
#     dataframe_regression.check(
#         summary.raw_fare_class_mix,
#         basename="raw_fare_class_mix",
#         default_tolerance=DEFAULT_TOLERANCE,
#     )
#
#
# @pytest.mark.filterwarnings("error")
# def test_3mkt_dp_demand_to_come(summary, dataframe_regression):
#     assert isinstance(summary, SummaryTables)
#     dataframe_regression.check(
#         summary.demand_to_come,
#         basename="demand_to_come",
#         default_tolerance=DEFAULT_TOLERANCE,
#     )


@pytest.mark.parametrize(
    "by_carrier, by_class",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
        ("AL1", False),
        ("AL1", True),
        (False, "Y5"),
        (True, "Y5"),
    ],
)
@pytest.mark.parametrize("measure", ["bookings", "revenue"])
def test_3mkt_dp_fig_segmentation_by_timeframe(summary, dataframe_regression, by_carrier, by_class, measure):
    assert isinstance(summary, SimulationTables)
    assert isinstance(
        summary.fig_segmentation_by_timeframe(measure, by_carrier=by_carrier, by_class=by_class),
        altair.TopLevelMixin,
    )
    dataframe_regression.check(
        summary.fig_segmentation_by_timeframe(
            measure, by_carrier=by_carrier, by_class=by_class, raw_df=True
        ).reset_index(drop=True),
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_dp_fig_carrier_load_factors(summary, dataframe_regression):
    assert isinstance(summary, SimulationTables)
    fig = summary.fig_carrier_load_factors()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_carrier_load_factors(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="fig_carrier_load_factors",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_dp_fig_carrier_mileage(summary, dataframe_regression):
    assert isinstance(summary, SimulationTables)
    fig = summary.fig_carrier_mileage()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_carrier_mileage(raw_df=True).reset_index(drop=True)
    # integer tests are flakey, change to float
    i = df.select_dtypes(include=["number"]).columns
    df[i] = df[i].astype("float") * 1.00000001
    dataframe_regression.check(
        df,
        basename="fig_carrier_mileage",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_dp_fig_carrier_revenues(summary, dataframe_regression):
    assert isinstance(summary, SimulationTables)
    fig = summary.fig_carrier_revenues()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_carrier_revenues(raw_df=True).reset_index(drop=True)
    # integer tests are flakey, change to float
    i = df.select_dtypes(include=["number"]).columns
    df[i] = df[i].astype("float") * 1.00000001
    dataframe_regression.check(
        df,
        basename="fig_carrier_revenues",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_dp_fig_carrier_yields(summary, dataframe_regression):
    assert isinstance(summary, SimulationTables)
    fig = summary.fig_carrier_yields()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_carrier_yields(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="fig_carrier_yields",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_dp_fig_fare_class_mix(summary, dataframe_regression):
    assert isinstance(summary, SimulationTables)
    fig = summary.fig_fare_class_mix()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_fare_class_mix(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="fig_fare_class_mix",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_dp_fig_od_fare_class_mix(summary, dataframe_regression):
    assert isinstance(summary, SimulationTables)
    fig = summary.fig_od_fare_class_mix(orig="BOS", dest="ORD")
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_od_fare_class_mix(orig="BOS", dest="ORD", raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="fig_od_fare_class_mix",
        default_tolerance=DEFAULT_TOLERANCE,
    )


@pytest.mark.parametrize("of", ["mu", "sigma"])
def test_3mkt_dp_fig_leg_forecasts(summary, dataframe_regression, of: Literal["mu", "sigma"]):
    assert isinstance(summary, SimulationTables)
    fig = summary.fig_leg_forecasts(of=of)
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_leg_forecasts(of=of, raw_df=True).reset_index(drop=True)
    dataframe_regression.check(df, default_tolerance=DEFAULT_TOLERANCE)
