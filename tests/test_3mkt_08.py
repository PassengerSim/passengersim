from typing import Literal

import altair
import numpy as np
import pytest
from pytest import approx

from passengersim import Simulation, demo_network
from passengersim.config import Config
from passengersim.database import Database
from passengersim.database.common_queries import leg_forecast_trace, leg_sales_trace
from passengersim.summary import SummaryTables

DEFAULT_TOLERANCE = dict(rtol=2e-02, atol=1e-06)


@pytest.fixture(scope="module")
def summary() -> SummaryTables:
    input_file = demo_network("3MKT/08-untrunc-em")
    cfg = Config.from_yaml(input_file)
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 500
    cfg.outputs.reports.add(("od_fare_class_mix", "BOS", "ORD"))
    sim = Simulation(cfg)
    summary = sim.run(summarizer=None)
    summary.sim = sim
    return summary


def test_3mkt_08_basic(summary):
    assert isinstance(summary, SummaryTables)
    assert isinstance(summary.cnx, Database)
    assert summary.cnx.is_open
    assert summary.cnx.engine == "sqlite"
    assert str(summary.cnx.filename) == ":memory:"
    assert isinstance(summary.sim, Simulation)
    assert summary.sim.sim.carriers[0].name == "AL1"
    assert summary.sim.sim.carriers[0].raw_load_factor_distribution() == approx(
        np.concat(
            (
                (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0),
                (0, 5, 2, 2, 1, 3, 6, 5, 2, 6, 3, 4, 6, 10, 4, 6, 10, 5, 7, 12),
                (4, 7, 13, 13, 15, 19, 14, 15, 30, 24, 25, 16, 21, 20, 20, 19),
                (24, 25, 40, 38, 29, 29, 32, 36, 36, 32, 35, 35, 47, 37, 39),
                (35, 36, 51, 34, 39, 33, 30, 448),
            )
        )
    )
    assert summary.sim.sim.legs[0].flt_no == 101
    # assert isinstance(summary.sim.sim.legs[0].carrier, Carrier)
    # assert summary.sim.sim.legs[0].carrier.name == "AL1"
    assert summary.sim.sim.legs[0].avg_load_factor() == approx(83.815)
    assert summary.sim.sim.legs[0].avg_local() == approx(41.82723856111675)

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


def test_3mkt_08_bookings_by_timeframe(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    dataframe_regression.check(
        summary.bookings_by_timeframe,
        basename="bookings_by_timeframe",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_08_segmentation_by_timeframe(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    dataframe_regression.check(
        summary.segmentation_by_timeframe.stack(0),
        basename="segmentation_by_timeframe",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_08_carriers(summary, dataframe_regression):
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


def test_3mkt_08_fare_class_mix(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    dataframe_regression.check(
        summary.fare_class_mix,
        basename="fare_class_mix",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_08_raw_fare_class_mix(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    dataframe_regression.check(
        summary.raw_fare_class_mix,
        basename="raw_fare_class_mix",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_08_demand_to_come(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    dataframe_regression.check(
        summary.demand_to_come,
        basename="demand_to_come",
        default_tolerance=DEFAULT_TOLERANCE,
    )


@pytest.mark.parametrize("leg_id", [101, 111])
@pytest.mark.parametrize("days_prior", [63, 7, 1])
@pytest.mark.parametrize("booking_class", ["Y0", "Y4", "Y5"])
def test_3mkt_08_forecast_traces(
    summary, leg_id, days_prior, booking_class, dataframe_regression
):
    assert isinstance(summary, SummaryTables)
    df = leg_forecast_trace(
        summary.cnx,
        leg_id=leg_id,
        booking_class=booking_class,
        days_prior=days_prior,
        burn_samples=0,
    )
    dataframe_regression.check(
        df,
        default_tolerance=DEFAULT_TOLERANCE,
    )


@pytest.mark.parametrize("leg_id", [101, 111])
@pytest.mark.parametrize("days_prior", [63, 7, 0])
@pytest.mark.parametrize("booking_class", ["Y0", "Y4", "Y5"])
def test_3mkt_08_sales_traces(
    summary, leg_id, days_prior, booking_class, dataframe_regression
):
    assert isinstance(summary, SummaryTables)
    df = leg_sales_trace(
        summary.cnx,
        leg_id=leg_id,
        booking_class=booking_class,
        days_prior=days_prior,
        burn_samples=0,
    )
    dataframe_regression.check(
        df,
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
def test_3mkt_08_fig_bookings_by_timeframe(
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


@pytest.mark.filterwarnings("error")
def test_3mkt_08_fig_carrier_load_factors(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    fig = summary.fig_carrier_load_factors()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_carrier_load_factors(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="fig_carrier_load_factors",
        default_tolerance=DEFAULT_TOLERANCE,
    )


@pytest.mark.filterwarnings("error")
def test_3mkt_08_fig_carrier_mileage(summary, dataframe_regression):
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


@pytest.mark.filterwarnings("error")
def test_3mkt_08_fig_carrier_revenues(summary, dataframe_regression):
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


@pytest.mark.filterwarnings("error")
def test_3mkt_08_fig_carrier_yields(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    fig = summary.fig_carrier_yields()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_carrier_yields(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="fig_carrier_yields",
        default_tolerance=DEFAULT_TOLERANCE,
    )


@pytest.mark.filterwarnings("error")
def test_3mkt_08_fig_fare_class_mix(summary, dataframe_regression):
    assert isinstance(summary, SummaryTables)
    fig = summary.fig_fare_class_mix()
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_fare_class_mix(raw_df=True).reset_index(drop=True)
    dataframe_regression.check(
        df,
        basename="fig_fare_class_mix",
        default_tolerance=DEFAULT_TOLERANCE,
    )


@pytest.mark.filterwarnings("error")
def test_3mkt_08_fig_od_fare_class_mix(summary, dataframe_regression):
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
def test_3mkt_08_fig_leg_forecasts(
    summary, dataframe_regression, of: Literal["mu", "sigma"]
):
    assert isinstance(summary, SummaryTables)
    fig = summary.fig_leg_forecasts(of=of)
    assert isinstance(fig, altair.TopLevelMixin)
    df = summary.fig_leg_forecasts(of=of, raw_df=True).reset_index(drop=True)
    dataframe_regression.check(df, default_tolerance=DEFAULT_TOLERANCE)


def sim_with_truncation_rule(truncation_rule=1) -> SummaryTables:
    input_file = demo_network("3MKT/08-untrunc-em")
    cfg = Config.from_yaml(input_file)
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 500
    cfg.outputs.reports.add(("od_fare_class_mix", "BOS", "ORD"))
    cfg.carriers["AL1"].truncation_rule = truncation_rule
    cfg.carriers["AL2"].truncation_rule = truncation_rule
    sim = Simulation(cfg)
    summary = sim.run(summarizer=None)
    summary.sim = sim
    return summary


@pytest.fixture(scope="module")
def summary1() -> SummaryTables:
    return sim_with_truncation_rule(1)


@pytest.fixture(scope="module")
def summary2() -> SummaryTables:
    return sim_with_truncation_rule(2)


@pytest.fixture(scope="module")
def summary3() -> SummaryTables:
    return sim_with_truncation_rule(3)


def test_3mkt_08_bookings_by_timeframe_1(summary1, dataframe_regression):
    assert isinstance(summary1, SummaryTables)
    dataframe_regression.check(
        summary1.bookings_by_timeframe,
        basename="bookings_by_timeframe_1",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_08_bookings_by_timeframe_2(summary2, dataframe_regression):
    assert isinstance(summary2, SummaryTables)
    dataframe_regression.check(
        summary2.bookings_by_timeframe,
        basename="bookings_by_timeframe_2",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_08_bookings_by_timeframe_3(summary3, dataframe_regression):
    assert isinstance(summary3, SummaryTables)
    dataframe_regression.check(
        summary3.bookings_by_timeframe,
        basename="bookings_by_timeframe_3",
        default_tolerance=DEFAULT_TOLERANCE,
    )


def test_3mkt_08_detrunc_bookings_by_timeframe(summary1, summary2, summary3):
    assert isinstance(summary3, SummaryTables)
    assert isinstance(summary1, SummaryTables)
    assert isinstance(summary2, SummaryTables)

    sim1 = summary1.sim
    sim2 = summary2.sim
    sim3 = summary3.sim
    assert len(sim1.paths) == 12
    for p in sim1.paths:
        if p.carrier_name == "AL1":
            assert p.truncation_rule == 1
        elif p.carrier_name == "AL2":
            assert p.truncation_rule == 1
        else:
            raise AssertionError(f"Unexpected carrier {p.carrier_name}")
    for p in sim2.paths:
        if p.carrier_name == "AL1":
            assert p.truncation_rule == 2
        elif p.carrier_name == "AL2":
            assert p.truncation_rule == 2
        else:
            raise AssertionError(f"Unexpected carrier {p.carrier_name}")
    for p in sim3.paths:
        if p.carrier_name == "AL1":
            assert p.truncation_rule == 3
        elif p.carrier_name == "AL2":
            assert p.truncation_rule == 3
        else:
            raise AssertionError(f"Unexpected carrier {p.carrier_name}")


def test_truncation_rule_1() -> SummaryTables:
    input_file = demo_network("3MKT/08-untrunc-em")
    cfg = Config.from_yaml(input_file)
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 500
    cfg.outputs.reports.add(("od_fare_class_mix", "BOS", "ORD"))
    cfg.carriers["AL1"].truncation_rule = 1
    cfg.carriers["AL2"].truncation_rule = 1
    sim = Simulation(cfg)

    sim.setup_scenario()
    sim.begin_trial(0)

    import numpy

    numpy.set_printoptions(linewidth=400)

    for s in range(11):
        with sim.run_single_sample():
            print("===== Sample", s, "=====")
            leg = sim.legs[101]
            bkt = leg.buckets[1]
            bs = {b.name: b.sold for b in leg.buckets}
            print(leg, bkt, "SOLD:", leg.sold, bs)

            if s < 9:
                continue

            for b in leg.buckets:
                v = b.forecast.get_vectors()
                # print(v.mean_in_timeframe[::-1])
                print(b.name, v.mean_to_departure)

            print(bkt.forecast.history.as_arrays()["sold"])
            print(bkt.forecast.history.as_arrays()["closed_flags"])
            print("FINAL TIMEFRAMES")
            print(bkt.forecast.get_detruncated_demand_array()[:, -1])
            print(bkt.forecast.get_detruncated_demand_array()[:, -2])
            print(bkt.forecast.get_detruncated_demand_array()[:, -3])
            print(bkt.forecast.get_detruncated_demand_array()[:, -4])

            # other_demand = np.asarray([
            #      [0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 3., 0., 0.],
            #      [2., 1., 1., 1., 0., 0., 0., 0., 2., 2., 0., 1., 1., 1., 4., 0.],
            #      [1., 0., 1., 1., 0., 1., 0., 0., 2., 0., 2., 3., 0., 1., 0., 0.],
            #      [1., 2., 0., 0., 0., 1., 0., 0., 0., 1., 0., 2., 1., 0., 1., 0.],
            #      [0., 0., 0., 0., 1., 2., 0., 0., 0., 2., 0., 5., 0., 0., 0., 0.],
            #      [0., 0., 0., 2., 1., 0., 0., 0., 0., 0., 0., 1., 0., 2., 2., 2.],
            #      [2., 0., 0., 0., 0., 1., 0., 0., 2., 0., 2., 2., 0., 0., 4., 1.],
            #      [0., 1., 1., 0., 1., 1., 1., 0., 0., 2., 1., 1., 0., 0., 2., 1.],
            #      [0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 2., 3., 0.],
            #      [1., 0., 0., 0., 1., 1., 0., 3., 2., 0., 1., 0., 0., 0., 0., 0.],
            # ])
            # print("OTHER DEMAND")
            # print(bkt.forecast.history.as_arrays()['sold'] - other_demand)
