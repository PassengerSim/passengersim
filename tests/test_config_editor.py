"""Tests for the copy-only interactive schedule editor."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path as FilePath
from zoneinfo import ZoneInfo

import pytest

import passengersim as pax
from passengersim.config.editor import ScheduleEditor, _shift_leg, edit_hub_schedule
from passengersim.config.legs import Leg
from passengersim.config.paths import Path


@pytest.fixture
def cfg():
    """Load a small two-carrier hub network."""
    return pax.Config.from_yaml(pax.demo_network("3MKT"))


def test_editor_uses_deep_copy_and_moves_only_edited_leg(cfg):
    """A schedule move must never mutate the supplied configuration."""
    source_dump = cfg.model_dump()
    source_leg = cfg.legs.select(leg_id=101)
    editor = edit_hub_schedule(cfg, "ORD")
    edited_leg = editor.edited_config.legs.select(leg_id=101)
    original_duration = edited_leg.arr_time - edited_leg.dep_time

    editor.move_leg(101, 15)

    assert editor.source_config is cfg
    assert editor.edited_config is not cfg
    assert edited_leg is not source_leg
    assert edited_leg.dep_time == source_leg.dep_time + 15 * 60
    assert edited_leg.arr_time == source_leg.arr_time + 15 * 60
    assert edited_leg.arr_time - edited_leg.dep_time == original_duration
    assert cfg.model_dump() == source_dump
    assert editor.changes[["leg_id", "delta_minutes"]].to_dict("records") == [{"leg_id": 101, "delta_minutes": 15}]


def test_inbound_and_outbound_rows_have_same_shift_semantics(cfg):
    """Inbound and outbound display orientation must not affect time edits."""
    editor = ScheduleEditor(cfg, "ORD")
    inbound = editor.edited_config.legs.select(leg_id=101)
    outbound = editor.edited_config.legs.select(leg_id=111)
    inbound_before = (inbound.dep_time, inbound.arr_time)
    outbound_before = (outbound.dep_time, outbound.arr_time)

    editor.move_leg(101, -20)
    editor.move_leg(111, 35)

    assert (inbound.dep_time, inbound.arr_time) == tuple(value - 20 * 60 for value in inbound_before)
    assert (outbound.dep_time, outbound.arr_time) == tuple(value + 35 * 60 for value in outbound_before)


def test_undo_redo_and_reset(cfg):
    """Editing history must restore complete leg timing states."""
    editor = ScheduleEditor(cfg, "ORD")
    leg = editor.edited_config.legs.select(leg_id=101)
    original = leg.model_dump()

    assert editor.undo() is False
    assert editor.redo() is False
    editor.move_leg(101, 10)
    changed = leg.model_dump()
    assert editor.undo() is True
    assert leg.model_dump() == original
    assert editor.redo() is True
    assert leg.model_dump() == changed

    editor.reset()

    assert leg.model_dump() == original
    assert editor.changes.empty
    assert editor.undo() is False


def test_drag_event_moves_exactly_one_leg(cfg):
    """A browser move event must dispatch one single-leg edit."""
    editor = ScheduleEditor(cfg, "ORD")
    before = {leg.leg_id: (leg.dep_time, leg.arr_time) for leg in editor.edited_config.legs}

    editor._event = {"type": "move", "leg_id": 101, "delta_seconds": 300, "nonce": "test"}

    after = {leg.leg_id: (leg.dep_time, leg.arr_time) for leg in editor.edited_config.legs}
    assert after[101] == tuple(value + 300 for value in before[101])
    assert {leg_id: times for leg_id, times in after.items() if leg_id != 101} == {
        leg_id: times for leg_id, times in before.items() if leg_id != 101
    }


def test_snap_event_persists_in_widget_settings(cfg):
    """A toolbar snap change must survive subsequent widget redraws."""
    editor = ScheduleEditor(cfg, "ORD")

    editor._event = {"type": "set_snap", "snap_minutes": 15, "nonce": "snap"}
    editor.move_leg(101, 15)

    assert editor.snap_minutes == 15
    assert editor._settings["snap_minutes"] == 15


def test_carrier_filter_limits_editable_legs(cfg):
    """Only filtered hub legs may be moved through an editor."""
    editor = ScheduleEditor(cfg, "ORD", carrier="AL1")

    assert {row["carrier"] for row in editor._legs_data} == {"AL1"}
    assert len(editor._legs_data) == 4
    with pytest.raises(KeyError, match="not visible"):
        editor.move_leg(201, 5)


def test_coloring_by_leg_tag_builds_stable_groups_and_legend(cfg):
    """Leg-tag coloring should use one deterministic color per tag value."""
    cfg.legs.select(leg_id=101).tags["fleet"] = "regional"
    cfg.legs.select(leg_id=102).tags["fleet"] = "regional"
    cfg.legs.select(leg_id=111).tags["fleet"] = "mainline"
    editor = ScheduleEditor(cfg, "ORD", coloring="leg.fleet")
    rows = {row["leg_id"]: row for row in editor._legs_data}

    assert rows[101]["color_group"] == "regional"
    assert rows[102]["color"] == rows[101]["color"]
    assert rows[111]["color_group"] == "mainline"
    assert rows[111]["color"] != rows[101]["color"]
    assert rows[201]["color_group"] == "(missing)"
    assert editor._settings["coloring"] == "leg.fleet"
    assert editor._settings["coloring_title"] == "Leg fleet"
    assert {item["label"] for item in editor._settings["legend"]} == {
        "regional",
        "mainline",
        "(missing)",
    }


def test_spoke_and_spoke_tag_coloring_match_hub_schedule_semantics(cfg):
    """Spoke selectors should categorize both inbound and outbound hub legs."""
    cfg.places["BOS"].tags["region"] = "east"
    cfg.places["LAX"].tags["region"] = "west"

    spoke_editor = ScheduleEditor(cfg, "ORD", coloring="spoke")
    region_editor = ScheduleEditor(cfg, "ORD", coloring="spoke.region")
    spokes = {row["leg_id"]: row["color_group"] for row in spoke_editor._legs_data}
    regions = {row["leg_id"]: row["color_group"] for row in region_editor._legs_data}

    assert spokes[101] == "BOS"
    assert spokes[111] == "LAX"
    assert regions[101] == "east"
    assert regions[111] == "west"


def test_default_and_explicit_carrier_coloring_are_equivalent(cfg):
    """The explicit carrier selector should preserve the default color scheme."""
    default_editor = ScheduleEditor(cfg, "ORD")
    carrier_editor = ScheduleEditor(cfg, "ORD", coloring="carrier")

    assert [(row["color_group"], row["color"]) for row in default_editor._legs_data] == [
        (row["color_group"], row["color"]) for row in carrier_editor._legs_data
    ]


def test_dense_defaults_fit_at_least_eighty_rows(cfg):
    """Default row geometry should display 80 legs without vertical scrolling."""
    editor = ScheduleEditor(cfg, "ORD")
    # The browser reserves 48 px above the first row and 8 px below the last.
    visible_capacity = 1 + (editor.max_height - 56) // editor.row_height

    assert editor.row_height == 6
    assert editor._settings["row_height"] == 6
    assert visible_capacity >= 80


def test_frontend_uses_an_instant_custom_hover_tooltip():
    """The frontend must not rely on the browser-delayed SVG title tooltip."""
    frontend = FilePath(inspect.getfile(ScheduleEditor)).with_name("_schedule_editor.js").read_text(encoding="utf-8")

    assert 'class="pse-tooltip"' in frontend
    assert 'hitLine.addEventListener("pointerenter"' in frontend
    assert "const rootBounds = root.getBoundingClientRect();" in frontend
    assert 'svgElement("title")' not in frontend


def test_factory_forwards_coloring_and_row_height(cfg):
    """The convenience function must expose the new display controls."""
    cfg.legs.select(leg_id=101).tags["fleet"] = "regional"

    editor = edit_hub_schedule(cfg, "ORD", coloring="leg.fleet", row_height=8)

    assert editor.coloring == "leg.fleet"
    assert editor.row_height == 8
    assert editor._settings["coloring"] == "leg.fleet"
    assert editor._settings["row_height"] == 8


def test_paths_remain_static_until_finish(cfg):
    """Ordinary schedule edits must not modify or regenerate paths."""
    cfg.paths.append(Path(path_id=9001, orig="BOS", dest="LAX", path_quality_index=1.0, legs=[101, 111]))
    editor = ScheduleEditor(cfg, "ORD")
    source_paths = [path.model_dump() for path in cfg.paths]
    edited_paths = [path.model_dump() for path in editor.edited_config.paths]

    editor.move_leg(101, 45)

    assert [path.model_dump() for path in cfg.paths] == source_paths
    assert [path.model_dump() for path in editor.edited_config.paths] == edited_paths
    row = next(row for row in editor._legs_data if row["leg_id"] == 101)
    assert row["connected_leg_ids"] == [101, 111]


def test_shift_leg_updates_midnight_metadata():
    """Cross-midnight moves must update date and arr_day without changing duration."""
    timezone = ZoneInfo("America/Los_Angeles")
    departure = datetime(2020, 3, 1, 23, 30, tzinfo=timezone)
    arrival = datetime(2020, 3, 2, 1, 30, tzinfo=timezone)
    leg = Leg(
        leg_id=1,
        carrier="AL1",
        fltno=1,
        orig="LAX",
        dest="SFO",
        orig_timezone=str(timezone),
        dest_timezone=str(timezone),
        dep_time=int(departure.timestamp()),
        arr_time=int(arrival.timestamp()),
        date=datetime(2020, 3, 1, tzinfo=UTC),
        arr_day=1,
        capacity=100,
        time_adjusted=True,
    )

    _shift_leg(leg, -2 * 3600)

    assert leg.dep_localtime.strftime("%Y-%m-%d %H:%M") == "2020-03-01 21:30"
    assert leg.arr_localtime.strftime("%Y-%m-%d %H:%M") == "2020-03-01 23:30"
    assert leg.date == datetime(2020, 3, 1, tzinfo=UTC)
    assert leg.arr_day == 0
    assert leg.duration_minutes == 120


def test_shift_leg_recomputes_dst_offsets():
    """Endpoint offsets must follow timezone rules when a move crosses DST."""
    timezone = "America/Los_Angeles"
    departure = datetime(2020, 3, 8, 8, 30, tzinfo=UTC)  # 00:30 PST
    arrival = departure + timedelta(hours=2)  # 03:30 PDT
    leg = Leg(
        leg_id=1,
        carrier="AL1",
        fltno=1,
        orig="LAX",
        dest="SFO",
        orig_timezone=timezone,
        dest_timezone=timezone,
        dep_time=int(departure.timestamp()),
        arr_time=int(arrival.timestamp()),
        date=datetime(2020, 3, 8, tzinfo=UTC),
        capacity=100,
        dep_time_offset=-8 * 3600,
        arr_time_offset=-7 * 3600,
        time_adjusted=True,
    )

    _shift_leg(leg, 2 * 3600)

    assert leg.dep_localtime.strftime("%H:%M %z") == "03:30 -0700"
    assert leg.arr_localtime.strftime("%H:%M %z") == "05:30 -0700"
    assert leg.dep_time_offset == -7 * 3600
    assert leg.arr_time_offset == -7 * 3600
    assert leg.duration_minutes == 120


def test_finish_rebuilds_connections_on_edited_copy_by_default(cfg, monkeypatch):
    """Finishing must replace paths on the copy and never pass the source config."""
    editor = ScheduleEditor(cfg, "ORD")
    editor.move_leg(101, 15)
    calls = []

    def fake_prebuild(config, *, inplace, **kwargs):
        calls.append((config, inplace, kwargs))
        config.paths.clear()
        config.paths.append(Path(path_id=77, orig="BOS", dest="LAX", path_quality_index=1.0, legs=[101, 111]))
        return config

    monkeypatch.setattr("passengersim.connection_builder.prebuild_connections", fake_prebuild)

    result = editor.finish()

    assert result is editor.edited_config
    assert result is not cfg
    assert calls == [(editor.edited_config, True, {"existing_paths": "replace"})]
    assert editor.finished is True
    assert editor.connections_rebuilt is True
    assert [path.path_id for path in editor.edited_config.paths] == [77]
    assert not cfg.paths
    with pytest.raises(RuntimeError, match="already been finalized"):
        editor.move_leg(101, 5)


def test_finish_can_explicitly_skip_connection_rebuild(cfg, monkeypatch):
    """The skip option must finish without invoking the connection builder."""
    editor = ScheduleEditor(cfg, "ORD")

    def unexpected_prebuild(*args, **kwargs):
        raise AssertionError("connection builder should have been skipped")

    monkeypatch.setattr("passengersim.connection_builder.prebuild_connections", unexpected_prebuild)

    result = editor.finish(rebuild_connections=False)

    assert result is editor.edited_config
    assert editor.finished is True
    assert editor.connections_rebuilt is False
    assert "without rebuilding" in editor._status["message"]


def test_failed_rebuild_leaves_editor_open(cfg, monkeypatch):
    """A rebuild error must be reported while preserving the ability to retry."""
    editor = ScheduleEditor(cfg, "ORD")

    def failed_prebuild(*args, **kwargs):
        raise ValueError("synthetic failure")

    monkeypatch.setattr("passengersim.connection_builder.prebuild_connections", failed_prebuild)

    with pytest.raises(ValueError, match="synthetic failure"):
        editor.finish()

    assert editor.finished is False
    assert editor._status["phase"] == "error"
    assert "synthetic failure" in editor._status["message"]
    editor.move_leg(101, 5)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"hub": "XXX"}, ValueError, "No place found"),
        ({"hub": "ORD", "snap_minutes": 0}, ValueError, "positive integer"),
        ({"hub": "ORD", "width": 300}, ValueError, "at least 480"),
        ({"hub": "ORD", "min_height": 300, "max_height": 200}, ValueError, "max_height"),
        ({"hub": "ORD", "row_height": 4}, ValueError, "from 5 through 24"),
        ({"hub": "ORD", "row_height": True}, ValueError, "from 5 through 24"),
        ({"hub": "ORD", "coloring": "leg."}, ValueError, "unrecognized coloring"),
        ({"hub": "ORD", "coloring": "unknown"}, ValueError, "unrecognized coloring"),
        ({"hub": "ORD", "coloring": 42}, TypeError, "string or None"),
        ({"hub": "ORD", "inplace": True}, TypeError, "does not support in-place"),
    ],
)
def test_editor_validates_constructor_arguments(cfg, kwargs, error, message):
    """Invalid display and schedule arguments should fail clearly."""
    with pytest.raises(error, match=message):
        ScheduleEditor(cfg, **kwargs)
