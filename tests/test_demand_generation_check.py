"""Regression tests for the demand-generation configuration check."""

import pytest

import passengersim as pax
from passengersim.config.checks.demand import _viz_segment_stats, _viz_volume, check_demand_generation


@pytest.fixture(scope="module")
def reversed_segment_result():
    """Run a small check whose demand order differs from its mean-WTP order."""
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))
    target_demands = [d for d in cfg.demands if d.orig == "BOS" and d.dest == "ORD"]
    assert [d.segment for d in target_demands] == ["business", "leisure"]

    target_demands.reverse()
    for demand in target_demands:
        demand.deterministic = True
    other_demands = [d for d in cfg.demands if not (d.orig == "BOS" and d.dest == "ORD")]
    cfg.demands = [*target_demands, *other_demands]

    return check_demand_generation(
        cfg,
        orig="BOS",
        dest="ORD",
        carrier="AL1",
        n_samples=3,
        n_draws_per_tf=50,
        wtp_resolution=10,
        raw_data=True,
    )


def test_check_demand_generation_preserves_accumulator_row_order(reversed_segment_result):
    """Segment labels must retain the order used to accumulate demand rows."""
    raw = reversed_segment_result

    assert raw.segments == ["leisure", "business"]
    assert raw.welford.mean.sum(axis=1) == pytest.approx([90.0, 70.0])
    assert raw.segment_stats.index.tolist() == ["business", "leisure"]


def test_demand_generation_charts_use_their_own_segment_orders(reversed_segment_result):
    """Volume rows use demand order while WTP statistics remain WTP-sorted."""
    raw = reversed_segment_result

    volume_spec = _viz_volume(raw, height=100, width=100).to_dict()
    volume_data = volume_spec["datasets"][volume_spec["data"]["name"]]
    volume_totals = {
        segment: sum(row["mean_demand"] for row in volume_data if row["segment"] == segment) for segment in raw.segments
    }
    assert volume_totals == pytest.approx({"leisure": 90.0, "business": 70.0})

    stats_spec = _viz_segment_stats(raw, height=100, width=100).to_dict()
    wtp_order = raw.segment_stats.index.tolist()
    assert stats_spec["layer"][0]["encoding"]["x"]["sort"] == wtp_order
    assert stats_spec["layer"][0]["encoding"]["color"]["sort"] == wtp_order
