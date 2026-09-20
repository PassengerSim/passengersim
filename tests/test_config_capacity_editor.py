"""Tests for the copy-only hub capacity and bank editor."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import passengersim as pax
from passengersim.config.editor import HubCapacityEditor, edit_hub_capacity


@pytest.fixture
def cfg():
    """Load a small network with two carriers sharing one hub."""
    return pax.Config.from_yaml(pax.demo_network("3MKT"))


def leg(editor: HubCapacityEditor, leg_id: int):
    """Select one leg from an editor's copied configuration."""
    return editor.edited_config.legs.select(leg_id=leg_id)


def test_carrier_is_a_required_nonempty_keyword(cfg):
    """Capacity editing must always have an explicit carrier scope."""
    with pytest.raises(TypeError, match="carrier"):
        edit_hub_capacity(cfg, "ORD")
    with pytest.raises(ValueError, match="carrier is required"):
        edit_hub_capacity(cfg, "ORD", carrier="")


def test_editor_deep_copies_and_exposes_only_carrier_hub_legs(cfg):
    """The source configuration and all hidden legs must remain independent."""
    source = cfg.model_dump()

    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")
    editor.set_capacity(101, 88)

    assert editor.source_config is cfg
    assert editor.edited_config is not cfg
    assert {row["leg_id"] for row in editor._legs_data} == {101, 102, 111, 112}
    assert {row["carrier"] for row in editor._legs_data} == {"AL1"}
    assert leg(editor, 101).capacity == 88
    assert cfg.model_dump() == source
    with pytest.raises(KeyError, match="not visible"):
        editor.set_capacity(201, 77)


def test_same_named_hidden_bank_members_are_untouched_and_excluded(cfg):
    """Bank operations and totals must be limited to visible carrier legs."""
    for leg_id in (101, 111, 201, 211):
        cfg.legs.select(leg_id=leg_id).tags["bank"] = "morning"

    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")
    editor.activate_bank("morning")

    assert editor.active_bank_totals == {
        "arrival_legs": 1,
        "arrival_capacity": 100,
        "departure_legs": 1,
        "departure_capacity": 120,
    }
    assert {row["leg_id"] for row in editor._legs_data if row["in_active_bank"]} == {101, 111}

    editor.rename_bank("peak")

    assert leg(editor, 101).tags["bank"] == "peak"
    assert leg(editor, 111).tags["bank"] == "peak"
    assert leg(editor, 201).tags["bank"] == "morning"
    assert leg(editor, 211).tags["bank"] == "morning"
    assert cfg.legs.select(leg_id=101).tags["bank"] == "morning"


def test_selecting_leg_activates_its_bank_and_unbanked_leg_clears_it(cfg):
    """Ordinary leg selection should drive bank highlighting and totals."""
    cfg.legs.select(leg_id=101).tags["bank"] = "alpha"
    cfg.legs.select(leg_id=111).tags["bank"] = "alpha"
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")

    editor.select_leg(111)

    assert editor.selected_leg_id == 111
    assert editor.active_bank == "alpha"
    assert editor.active_bank_totals["departure_capacity"] == 120

    editor.select_leg(102)

    assert editor.selected_leg_id == 102
    assert editor.active_bank is None
    assert editor.active_bank_totals["arrival_capacity"] == 0


def test_staged_bank_membership_is_mutually_exclusive_but_not_exhaustive(cfg):
    """Saving a bank should overwrite visible prior membership only for selected legs."""
    cfg.legs.select(leg_id=101).tags["bank"] = "old"
    cfg.legs.select(leg_id=201).tags["bank"] = "new"
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")

    editor.new_bank("new")
    editor.toggle_bank_leg(101)
    editor.toggle_bank_leg(111)

    assert editor.active_bank_totals == {
        "arrival_legs": 1,
        "arrival_capacity": 100,
        "departure_legs": 1,
        "departure_capacity": 120,
    }
    assert leg(editor, 101).tags["bank"] == "old"

    editor.save_bank()

    assert leg(editor, 101).tags["bank"] == "new"
    assert leg(editor, 111).tags["bank"] == "new"
    assert "bank" not in leg(editor, 102).tags
    assert "bank" not in leg(editor, 112).tags
    assert leg(editor, 201).tags["bank"] == "new"


def test_editing_existing_bank_removes_deselected_visible_members_only(cfg):
    """A committed membership edit should use the staged visible set exactly."""
    for leg_id in (101, 102, 201):
        cfg.legs.select(leg_id=leg_id).tags["cluster"] = "A"
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1", bank_tag="cluster")
    editor.activate_bank("A")
    editor.begin_bank_edit()
    editor.toggle_bank_leg(102)
    editor.toggle_bank_leg(111)
    editor.save_bank()

    assert leg(editor, 101).tags["cluster"] == "A"
    assert "cluster" not in leg(editor, 102).tags
    assert leg(editor, 111).tags["cluster"] == "A"
    assert leg(editor, 201).tags["cluster"] == "A"


def test_canceling_new_bank_discards_all_staged_membership(cfg):
    """Cancel must leave configured tags unchanged and clear a transient bank."""
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")
    editor.new_bank("temporary")
    editor.toggle_bank_leg(101)

    editor.cancel_bank_edit()

    assert editor.active_bank is None
    assert editor.banks == ()
    assert all("bank" not in leg(editor, leg_id).tags for leg_id in (101, 102, 111, 112))


def test_scalar_capacity_presets_are_sorted_and_frozen(cfg):
    """Carrier capacity suggestions should be ascending and fixed at initialization."""
    cfg.legs.select(leg_id=101).capacity = 180
    cfg.legs.select(leg_id=102).capacity = 90
    cfg.legs.select(leg_id=111).capacity = 120
    cfg.legs.select(leg_id=112).capacity = 0

    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")

    assert editor.common_capacities == (90, 120, 180)
    assert editor._settings["common_capacities"] == [90, 120, 180]

    editor.set_capacity(101, 333)

    assert editor.common_capacities == (90, 120, 180)
    assert editor._settings["common_capacities"] == [90, 120, 180]


def test_scalar_capacity_requires_a_nonnegative_integer(cfg):
    """Scalar legs should accept zero but reject dictionaries, booleans, and negatives."""
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")

    editor.set_capacity(101, 0)
    assert leg(editor, 101).capacity == 0

    with pytest.raises(TypeError, match="require an integer"):
        editor.set_capacity(101, {"Y": 10})
    with pytest.raises(ValueError, match="nonnegative integer"):
        editor.set_capacity(101, True)
    with pytest.raises(ValueError, match="nonnegative integer"):
        editor.set_capacity(101, -1)


def test_finish_removes_visible_zero_capacity_leg_but_preserves_hidden_zero_leg(cfg):
    """Finalization should delete zero-capacity legs only inside the editor scope."""
    cfg.legs.select(leg_id=201).capacity = 0
    source = cfg.model_dump()
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")

    editor.set_capacity(101, 0)

    assert leg(editor, 101).capacity == 0
    assert next(row for row in editor._legs_data if row["leg_id"] == 101)["pending_deletion"] is True
    assert "Marked leg 101 for deletion" in editor._status["message"]

    result = editor.finish()
    result_leg_ids = {item.leg_id for item in result.legs}

    assert 101 not in result_leg_ids
    assert 201 in result_leg_ids
    assert result.legs.select(leg_id=201).capacity == 0
    assert cfg.model_dump() == source
    assert "removed 1 zero-capacity leg" in editor._status["message"]


def test_finish_removes_preexisting_visible_zero_capacity_leg(cfg):
    """A visible leg that starts at zero should also be absent from final output."""
    cfg.legs.select(leg_id=101).capacity = 0
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")

    result = editor.finish()

    assert 101 not in {item.leg_id for item in result.legs}
    assert cfg.legs.select(leg_id=101).capacity == 0


def test_zero_capacity_deletion_can_be_undone_before_finish(cfg):
    """A pending deletion should remain an ordinary reversible capacity edit."""
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")
    editor.set_capacity(101, 0)

    assert editor.undo() is True
    result = editor.finish()

    assert result.legs.select(leg_id=101).capacity == 100


def test_finish_removes_all_zero_cabin_leg_and_keeps_partial_cabin_leg(cfg):
    """Cabin legs should be deleted only when their total capacity is zero."""
    cfg.legs.select(leg_id=101).capacity = {"Y": 90, "J": 10}
    cfg.legs.select(leg_id=102).capacity = {"Y": 90, "J": 10}
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")

    editor.set_capacity(101, {"Y": 0, "J": 0})
    editor.set_capacity(102, {"Y": 80, "J": 0})
    result = editor.finish()

    result_leg_ids = {item.leg_id for item in result.legs}
    assert 101 not in result_leg_ids
    assert result.legs.select(leg_id=102).capacity == {"Y": 80, "J": 0}


def test_multi_cabin_capacity_has_separate_validated_fields(cfg):
    """A multi-cabin leg must preserve cabin keys and edit each cabin independently."""
    cfg.legs.select(leg_id=101).capacity = {"Y": 90, "J": 10}
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")
    row = next(row for row in editor._legs_data if row["leg_id"] == 101)

    assert row["capacity_kind"] == "cabins"
    assert row["capacity"] == {"Y": 90, "J": 10}

    editor.set_capacity(101, {"Y": 80, "J": 12})

    assert leg(editor, 101).capacity == {"Y": 80, "J": 12}
    assert leg(editor, 101).total_capacity == 92
    with pytest.raises(TypeError, match="dictionary"):
        editor.set_capacity(101, 92)
    with pytest.raises(ValueError, match="preserve"):
        editor.set_capacity(101, {"Y": 92})
    with pytest.raises(ValueError, match="nonnegative integer"):
        editor.set_capacity(101, {"Y": 80, "J": True})


def test_capacity_changes_recalculate_active_directional_totals(cfg):
    """An active bank's arrival and departure totals should update immediately."""
    cfg.legs.select(leg_id=101).tags["bank"] = "A"
    cfg.legs.select(leg_id=111).tags["bank"] = "A"
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")
    editor.select_leg(101)

    editor.set_capacity(101, 75)
    editor.set_capacity(111, 135)

    assert editor.active_bank_totals["arrival_capacity"] == 75
    assert editor.active_bank_totals["departure_capacity"] == 135
    assert editor._state["totals"] == editor.active_bank_totals


def test_history_and_reset_restore_capacity_and_bank_together(cfg):
    """Undo, redo, and reset should restore complete visible editable state."""
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")
    editor.set_capacity(101, 85)
    editor.new_bank("A")
    editor.toggle_bank_leg(101)
    editor.save_bank()

    assert leg(editor, 101).capacity == 85
    assert leg(editor, 101).tags["bank"] == "A"
    assert editor.undo() is True
    assert "bank" not in leg(editor, 101).tags
    assert editor.undo() is True
    assert leg(editor, 101).capacity == 100
    assert editor.redo() is True
    assert leg(editor, 101).capacity == 85

    editor.reset()

    assert leg(editor, 101).capacity == 100
    assert "bank" not in leg(editor, 101).tags
    assert editor.changes.empty


def test_changes_reports_bank_capacity_and_total_deltas(cfg):
    """The audit table should describe only changed visible legs."""
    cfg.legs.select(leg_id=101).capacity = {"Y": 90, "J": 10}
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")
    editor.set_capacity(101, {"Y": 80, "J": 8})

    assert editor.changes.to_dict("records") == [
        {
            "leg_id": 101,
            "carrier": "AL1",
            "fltno": 101,
            "orig": "BOS",
            "dest": "ORD",
            "original_bank": None,
            "edited_bank": None,
            "original_capacity": {"Y": 90, "J": 10},
            "edited_capacity": {"Y": 80, "J": 8},
            "original_total_capacity": 100,
            "edited_total_capacity": 88,
        }
    ]


def test_coloring_matches_schedule_editor_arguments(cfg):
    """Leg and spoke coloring selectors should populate stable groups and legends."""
    cfg.legs.select(leg_id=101).tags["fleet"] = "regional"
    cfg.legs.select(leg_id=102).tags["fleet"] = "regional"

    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1", coloring="leg.fleet")
    rows = {row["leg_id"]: row for row in editor._legs_data}

    assert rows[101]["color_group"] == "regional"
    assert rows[101]["color"] == rows[102]["color"]
    assert rows[111]["color_group"] == "(missing)"
    assert editor._settings["coloring_title"] == "Leg fleet"


def test_finish_returns_copy_without_rebuilding_connections(cfg, monkeypatch):
    """Capacity finalization must neither rebuild paths nor expose further edits."""
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("capacity editor must not rebuild connections")

    monkeypatch.setattr("passengersim.connection_builder.prebuild_connections", fail_if_called)
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")
    editor.set_capacity(101, 88)

    result = editor.finish()

    assert result is editor.edited_config
    assert result is not cfg
    assert editor.finished is True
    assert called is False
    with pytest.raises(RuntimeError, match="already been finalized"):
        editor.set_capacity(101, 77)


def test_finish_rejects_unsaved_bank_membership(cfg):
    """Users must explicitly save or cancel staged membership before finishing."""
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")
    editor.new_bank("A")

    with pytest.raises(RuntimeError, match="save or cancel"):
        editor.finish()


def test_factory_has_no_inplace_mode(cfg):
    """Neither class nor factory should offer an in-place escape hatch."""
    with pytest.raises(TypeError, match="inplace"):
        edit_hub_capacity(cfg, "ORD", carrier="AL1", inplace=True)
    with pytest.raises(TypeError, match="in-place"):
        HubCapacityEditor(cfg, "ORD", carrier="AL1", inplace=True)


def test_frontend_has_instant_tooltip_and_capacity_specific_controls():
    """The browser implementation should satisfy the interactive UI contract."""
    source_dir = Path(inspect.getfile(HubCapacityEditor)).parent
    frontend = source_dir.joinpath("_capacity_editor.js").read_text(encoding="utf-8")
    stylesheet = source_dir.joinpath("_schedule_editor.css").read_text(encoding="utf-8")

    assert 'class="pse-tooltip"' in frontend
    assert 'hitLine.addEventListener("pointerenter"' in frontend
    assert "const rootBounds = root.getBoundingClientRect();" in frontend
    assert 'leg.capacity_kind === "cabins"' in frontend
    assert "Object.entries(leg.capacity)" in frontend
    assert "settings.common_capacities" in frontend
    assert 'deleteLeg.textContent = "Delete leg"' in frontend
    assert 'className = "pce-delete-leg"' in frontend
    assert 'classes.push("pce-pending-deletion")' in frontend
    assert "Object.keys(leg.capacity).map((cabin) => [cabin, 0])" in frontend
    assert "Will be removed when finished" in frontend
    assert '"toggle_bank_leg"' in frontend
    assert "const onStatus = () => {\n    renderControls();\n    renderChart();" in frontend
    assert 'svgElement("title")' not in frontend
    assert (
        "position: relative;"
        in stylesheet.split(".passengersim-schedule-editor .pse-shell {", maxsplit=1)[1].split("}", maxsplit=1)[0]
    )
    assert (
        "position: absolute;"
        in stylesheet.split(".passengersim-capacity-editor .pce-capacity-popover {", maxsplit=1)[1].split(
            "}", maxsplit=1
        )[0]
    )


def test_dense_defaults_fit_at_least_eighty_rows(cfg):
    """Capacity editor defaults should preserve the dense schedule geometry."""
    editor = edit_hub_capacity(cfg, "ORD", carrier="AL1")
    visible_capacity = 1 + (editor.max_height - 56) // editor.row_height

    assert editor.row_height == 6
    assert visible_capacity >= 80
