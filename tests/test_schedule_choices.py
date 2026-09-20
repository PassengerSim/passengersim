"""Tests for schedule-choice diagnostics."""

import altair as alt
import pandas as pd
import pytest
from passengersim_core import Carrier, ChoiceModel, Fare, Leg, Market, Path

import passengersim as pax
from passengersim.config.checks import check_schedule_choices
from passengersim.config.todd_curves import ToddCurve


def _make_core_demand():
    """Create a two-carrier demand and retain its weakly referenced market."""
    market = Market("AAA", "BBB")
    demand = market.add_demand(segment="business", base_demand=100)
    demand.reference_price = 500.0
    demand.choice_model = ChoiceModel("business", "pods")

    carriers = [Carrier("AA"), Carrier("BB")]
    paths = []
    for leg_id, (carrier, hour) in enumerate(zip(carriers, [8, 10]), start=1):
        leg = Leg(
            leg_id,
            carrier,
            100 + leg_id,
            "AAA",
            "BBB",
            100,
            dep_time=hour * 3600,
            arr_time=(hour + 1) * 3600,
        )
        path = Path("AAA", "BBB", 100.0, carrier)
        path.add_leg(leg)
        demand.add_path(path)
        paths.append(path)

    fares = [
        Fare(carriers[0], "AAA", "BBB", "L", 90.0),
        Fare(carriers[0], "AAA", "BBB", "M", 125.0),
        Fare(carriers[0], "AAA", "BBB", "Y", 150.0),
        Fare(carriers[1], "AAA", "BBB", "Y", 200.0),
    ]
    fares[1].adv_purch = 7
    for fare in fares:
        demand.add_fare(fare)

    # CoreDemand holds a weak market reference, so callers must retain these
    # wrappers for the duration of each test.
    keepalive = market, carriers, paths, fares
    return demand, keepalive


def test_check_schedule_choices_selects_lowest_eligible_fare_per_carrier():
    """Floor and advance-purchase filters apply before carrier minima."""
    demand, _keepalive = _make_core_demand()

    result = check_schedule_choices(
        demand,
        floor_price=100.0,
        adv_purch=5,
        n_draws=500,
        raw_df=True,
    )

    assert list(result.columns) == [
        "path_id",
        "carrier",
        "routing",
        "booking_class",
        "price",
        "departure",
        "arrival",
        "duration",
        "choices",
        "purchase_probability",
        "choice_share",
    ]
    assert list(result["carrier"]) == ["AA", "BB"]
    assert list(result["booking_class"]) == ["Y", "Y"]
    assert list(result["price"]) == [150.0, 200.0]
    assert list(result["departure"].dt.hour) == [8, 10]
    assert list(result["arrival"].dt.hour) == [9, 11]
    assert list(result["duration"]) == [60.0, 60.0]
    assert list(result["routing"]) == ["AAA-BBB", "AAA-BBB"]
    assert result["choices"].sum() <= 500
    assert result["purchase_probability"].sum() == pytest.approx(result["choices"].sum() / 500)
    assert result["choice_share"].sum() == pytest.approx(1.0)


def test_check_schedule_choices_can_filter_carrier():
    """A carrier filter removes both other paths and other fares."""
    demand, _keepalive = _make_core_demand()

    result = check_schedule_choices(demand, carrier="BB", n_draws=50, raw_df=True)

    assert len(result) == 1
    assert result.loc[0, "carrier"] == "BB"
    assert result.loc[0, "price"] == 200.0


def test_check_schedule_choices_rejects_invalid_input_or_empty_offer_set():
    """Invalid inputs and filters that remove every offer fail clearly."""
    demand, _keepalive = _make_core_demand()

    with pytest.raises(TypeError, match="core.Demand or Config"):
        check_schedule_choices(object())
    with pytest.raises(ValueError, match="floor_price must be"):
        check_schedule_choices(demand, floor_price=float("nan"))
    with pytest.raises(ValueError, match=r"floor_price_by_path\[-1\] must be"):
        check_schedule_choices(demand, floor_price_by_path={-1: float("inf")})
    with pytest.raises(ValueError, match="No eligible offers"):
        check_schedule_choices(demand, floor_price=1_000.0)


def test_check_schedule_choices_returns_faceted_chart():
    """Chart mode plots conditional choice share against local departure."""
    demand, _keepalive = _make_core_demand()

    result = check_schedule_choices(demand, n_draws=50)
    spec = result.to_dict()

    assert isinstance(result, alt.FacetChart)
    assert spec["facet"]["row"]["field"] == "carrier"
    assert spec["spec"]["encoding"]["x"]["field"] == "departure"
    assert spec["spec"]["encoding"]["y"]["field"] == "choice_share"
    tooltip_fields = {tooltip["field"] for tooltip in spec["spec"]["encoding"]["tooltip"]}
    assert {"arrival", "duration"} <= tooltip_fields


def test_check_schedule_choices_config_uses_real_schedule_and_all_carriers():
    """Config mode must not replace a schedule with one synthetic itinerary."""
    config = pax.Config.from_yaml(pax.demo_network("3MKT"))
    config.todd_curves = {
        "morning": ToddCurve(name="morning", probabilities={8: 1.0}),
    }
    for demand in config.demands:
        demand.todd_curve = "morning"
    original_demand_count = len(config.demands)

    result = check_schedule_choices(
        config,
        orig="BOS",
        dest="ORD",
        segment="business",
        adv_purch=5,
        n_draws=100,
        raw_df=True,
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result["carrier"]) == {"AL1", "AL2"}
    assert set(result["departure"].dt.hour) == {8, 14}
    assert len(result) == 4
    assert len(config.demands) == original_demand_count


def test_check_schedule_choices_path_floor_prices_override_global_default():
    """Per-path floors may be lower or higher while omitted paths use the global floor."""
    config = pax.Config.from_yaml(pax.demo_network("3MKT"))
    baseline = check_schedule_choices(
        config,
        orig="BOS",
        dest="ORD",
        segment="business",
        floor_price=200.0,
        n_draws=10,
        raw_df=True,
    )
    lower_floor_path = baseline.loc[0, "path_id"]
    higher_floor_path = baseline.loc[1, "path_id"]

    result = check_schedule_choices(
        config,
        orig="BOS",
        dest="ORD",
        segment="business",
        floor_price=200.0,
        floor_price_by_path={lower_floor_path: 0.0, higher_floor_path: 350.0},
        n_draws=10,
        raw_df=True,
    ).set_index("path_id")

    assert result.loc[lower_floor_path, "price"] == 100.0
    assert result.loc[higher_floor_path, "price"] == 400.0
    assert (result.drop(index=[lower_floor_path, higher_floor_path])["price"] == 200.0).all()
