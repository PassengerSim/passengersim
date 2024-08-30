from typing import Literal

import altair
import pytest

from passengersim import Simulation, demo_network
from passengersim.config import Config
from passengersim.summary import SummaryTables

DEFAULT_TOLERANCE = dict(rtol=2e-02, atol=1e-06)


@pytest.fixture(scope="module")
def summary() -> SummaryTables:
    input_file = demo_network("3MKT/12-udp")
    cfg = Config.from_yaml(input_file)
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 500
    sim = Simulation(cfg)
    summary = sim.run()
    return summary


def test_3mkt_dp_bookings_by_timeframe(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    dataframe_regression.check(
        summary.bookings_by_timeframe,
        basename="bookings_by_timeframe",
        default_tolerance=DEFAULT_TOLERANCE,
    )


@pytest.mark.filterwarnings("error")
def test_3mkt_dp_carriers(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    df = summary.carriers
    # integer tests are flakey, change to float
    i = df.select_dtypes(include=["number"]).columns
    df[i] = df[i].astype("float") * 1.00000001
    dataframe_regression.check(
        df,
        basename="carriers",
        default_tolerance=DEFAULT_TOLERANCE,
    )


@pytest.mark.filterwarnings("error")
def test_3mkt_dp_fare_class_mix(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    dataframe_regression.check(
        summary.fare_class_mix,
        basename="fare_class_mix",
        default_tolerance=DEFAULT_TOLERANCE,
    )


@pytest.mark.filterwarnings("error")
def test_3mkt_dp_raw_fare_class_mix(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    dataframe_regression.check(
        summary.raw_fare_class_mix,
        basename="raw_fare_class_mix",
        default_tolerance=DEFAULT_TOLERANCE,
    )


@pytest.mark.filterwarnings("error")
def test_3mkt_dp_demand_to_come(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    dataframe_regression.check(
        summary.demand_to_come,
        basename="demand_to_come",
        default_tolerance=DEFAULT_TOLERANCE,
    )


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
def test_3mkt_dp_fig_bookings_by_timeframe(
    summary, dataframe_regression, by_carrier, by_class
):
    assert isinstance(summary, SummaryTables)
    assert isinstance(
        summary.fig_bookings_by_timeframe(by_carrier=by_carrier, by_class=by_class),
        altair.TopLevelMixin,
    )
    dataframe_regression.check(
        summary.fig_bookings_by_timeframe(
            by_carrier=by_carrier, by_class=by_class, raw_df=True
        ).reset_index(drop=True),
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_dp_fig_carrier_load_factors(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    fig = summary.fig_carrier_load_factors()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_carrier_load_factors(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="fig_carrier_load_factors",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_dp_fig_carrier_mileage(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
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
    assert isinstance(summary, SummaryTables)
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
    assert isinstance(summary, SummaryTables)
    fig = summary.fig_carrier_yields()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_carrier_yields(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="fig_carrier_yields",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_dp_fig_fare_class_mix(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    fig = summary.fig_fare_class_mix()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_fare_class_mix(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="fig_fare_class_mix",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_dp_fig_od_fare_class_mix(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    fig = summary.fig_od_fare_class_mix(orig="BOS", dest="ORD")
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_od_fare_class_mix(orig="BOS", dest="ORD", raw_df=True).reset_index(
        drop=True
    )
    dataframe_regression.check(
        df,
        basename="fig_od_fare_class_mix",
        default_tolerance=DEFAULT_TOLERANCE,
    )


@pytest.mark.parametrize("of", ["mu", "sigma"])
def test_3mkt_dp_fig_leg_forecasts(
    summary, dataframe_regression, of: Literal["mu", "sigma"]
):
    assert isinstance(summary, SummaryTables)
    fig = summary.fig_leg_forecasts(of=of)
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_leg_forecasts(of=of, raw_df=True).reset_index(drop=True)
    dataframe_regression.check(df, default_tolerance=DEFAULT_TOLERANCE)
