import datetime
import zoneinfo

import altair
import pytest
from pytest import approx

from passengersim import Simulation, demo_network
from passengersim.config import Config
from passengersim.database import Database
from passengersim.rm.specialty_systems.fcfs import FirstComeFirstServed  # noqa: F401
from passengersim.summaries import SimulationTables

DEFAULT_TOLERANCE = dict(rtol=2e-02, atol=1e-06)


@pytest.fixture(scope="module")
def summary() -> SimulationTables:
    input_file = demo_network("3mkt/01-base")
    cfg = Config.from_yaml(input_file)
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 500
    cfg.simulation_controls.allow_unused_restrictions = True
    cfg.outputs.reports.add(("od_fare_class_mix", "BOS", "ORD"))
    cfg.outputs.reports.add("load_factor_distribution")
    cfg.outputs._write_no_files()
    sim = Simulation(cfg)
    summary = sim.run()
    summary.sim = sim
    return summary


def test_3mkt_01_time_zones():
    input_file = demo_network("3MKT/01-base")
    cfg = Config.from_yaml(input_file)
    assert len(cfg.legs) == 8
    leg = cfg.legs[0]
    assert leg.orig == "BOS"
    assert leg.dest == "ORD"
    assert leg.orig_timezone == "America/New_York"
    assert leg.dest_timezone == "America/Chicago"
    assert leg.dep_localtime == datetime.datetime(2020, 3, 1, 8, 0, tzinfo=zoneinfo.ZoneInfo(key="America/New_York"))
    assert leg.arr_localtime == datetime.datetime(2020, 3, 1, 10, 0, tzinfo=zoneinfo.ZoneInfo(key="America/Chicago"))
    assert leg.dep_time == 1583067600
    assert leg.arr_time == 1583078400
    assert leg.distance == approx(863.753282)


def test_3mkt_01_basic(summary, num_regression):
    assert isinstance(summary, SimulationTables)
    assert isinstance(summary.cnx, Database)
    # assert summary.cnx.is_open
    assert summary.cnx.engine == "sqlite"
    # assert str(summary.cnx.filename) == ":memory:"
    assert isinstance(summary.sim, Simulation)
    assert summary.sim.eng.carriers[0].name == "AL1"
    num_regression.check(
        {
            "raw_load_factor_distribution": summary.sim.eng.carriers[0].raw_load_factor_distribution(),
        },
        basename="load_factor_distribution",
    )
    assert summary.sim.eng.legs[0].flt_no == 101
    num_regression.check(
        {
            "avg_load_factor": summary.sim.eng.legs[0].avg_load_factor(),
            "avg_local": summary.sim.eng.legs[0].avg_local(),
        },
        basename="load_factor_data",
    )

    # for each leg, check that the sum of gt_sold for all paths equals the leg's gt_sold
    for leg in summary.sim.eng.legs:
        local_sold, total_sold = 0, 0
        for pth in summary.sim.eng.paths:
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


def test_3mkt_01_segmentation_by_timeframe(summary, dataframe_regression):
    dataframe_regression.check(
        summary.segmentation_by_timeframe.stack(0, future_stack=False),
        basename="segmentation_by_timeframe",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_01_carriers(summary, dataframe_regression):
    df = summary.carriers
    # integer tests are flakey, change to float
    i = df.select_dtypes(include=["number"]).columns
    df[i] = df[i].astype("float") * 1.00000001
    dataframe_regression.check(
        df,
        basename="carriers",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_01_leg_local_dist(summary, dataframe_regression):
    df = summary.fig_leg_local_share_distribution(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="leg_local_fraction_distribution",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_01_leg_forecasts(summary, dataframe_regression):
    dataframe_regression.check(
        summary.leg_forecasts,
        basename="leg_forecasts",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_01_fare_class_mix(summary, dataframe_regression):
    dataframe_regression.check(
        summary.fare_class_mix,
        basename="fare_class_mix",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_01_demand_to_come(summary, dataframe_regression):
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
def test_3mkt_01_fig_bookings_by_timeframe(summary, dataframe_regression, by_carrier, by_class):
    assert isinstance(
        summary.fig_bookings_by_timeframe(by_carrier=by_carrier, by_class=by_class),
        altair.TopLevelMixin,
    )
    dataframe_regression.check(
        summary.fig_bookings_by_timeframe(by_carrier=by_carrier, by_class=by_class, raw_df=True).reset_index(drop=True),
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_01_fig_carrier_load_factors(summary, dataframe_regression):
    fig = summary.fig_carrier_load_factors()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_carrier_load_factors(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="fig_carrier_load_factors",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_01_fig_carrier_mileage(summary, dataframe_regression):
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


def test_3mkt_01_fig_carrier_revenues(summary, dataframe_regression):
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


def test_3mkt_01_fig_carrier_yields(summary, dataframe_regression):
    fig = summary.fig_carrier_yields()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_carrier_yields(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="fig_carrier_yields",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_01_fig_fare_class_mix(summary, dataframe_regression):
    fig = summary.fig_fare_class_mix()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_fare_class_mix(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="fig_fare_class_mix",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_01_fig_od_fare_class_mix(summary, dataframe_regression):
    fig = summary.fig_od_fare_class_mix(orig="BOS", dest="ORD")
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_od_fare_class_mix(orig="BOS", dest="ORD", raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="fig_od_fare_class_mix",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_01_fig_leg_forecasts(summary, dataframe_regression):
    fig = summary.fig_leg_forecasts()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_leg_forecasts(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(df, basename="fig_leg_forecasts", default_tolerance=DEFAULT_TOLERANCE)


@pytest.mark.parametrize("by_carrier", [False, True, "AL1"])
@pytest.mark.filterwarnings("error")
def test_3mkt_01_fig_load_factor_grouped(summary, dataframe_regression, by_carrier):
    fig = summary.fig_leg_load_factor_distribution(by_carrier=by_carrier)
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_leg_load_factor_distribution(by_carrier=by_carrier, raw_df=True).reset_index(drop=True)
    dataframe_regression.check(df, default_tolerance=DEFAULT_TOLERANCE)
