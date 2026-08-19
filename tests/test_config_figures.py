"""Regression tests for charts built from configuration data."""

import pytest

import passengersim as pax


@pytest.fixture(scope="module")
def cfg():
    """Load a small configuration shared by the figure tests."""
    return pax.Config.from_yaml(pax.demo_network("3MKT"))


@pytest.mark.parametrize("figure_name", ["fig_speed_by_distance", "fig_duration_by_distance"])
def test_leg_distance_figure_serializes_leg_ids(cfg, figure_name):
    """The indexed leg ID must be present in the chart tooltip data."""
    spec = getattr(cfg, figure_name)().to_dict()

    assert spec["encoding"]["tooltip"][0]["field"] == "leg_id"
    data = spec["datasets"][spec["data"]["name"]]
    assert data
    assert all("leg_id" in row for row in data)
