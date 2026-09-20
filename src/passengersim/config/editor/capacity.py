"""Interactive tools for editing hub banks and flight-leg capacity."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from traitlets import Dict, Int, List, observe

try:
    import anywidget
except ImportError as err:  # pragma: no cover - exercised in installations without the lab extra
    raise ImportError(
        "The capacity editor requires Jupyter widget dependencies. Install PassengerSim with the 'lab' extra."
    ) from err

from passengersim.config.base import Config

from ._hub import color_scale, leg_color_group, parse_coloring

__all__ = ["HubCapacityEditor", "edit_hub_capacity"]

CapacityValue = int | dict[str, int]


@dataclass(frozen=True)
class _LegState:
    """Editable capacity-editor state for one leg."""

    capacity: CapacityValue
    bank: str | None


@dataclass(frozen=True)
class _CapacityEdit:
    """One reversible capacity or bank operation."""

    label: str
    before: dict[int, _LegState]
    after: dict[int, _LegState]


def _validate_capacity_integer(value: Any, *, label: str = "capacity") -> int:
    """Validate one scalar or cabin capacity value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _normalize_bank_name(value: Any) -> str:
    """Validate and normalize a user-entered bank name."""
    if not isinstance(value, str):
        raise TypeError("bank name must be a string")
    result = value.strip()
    if not result:
        raise ValueError("bank name must not be empty")
    return result


class HubCapacityEditor(anywidget.AnyWidget):
    """Edit banks and capacities for one carrier's legs at a hub.

    The supplied configuration is always deep-copied. Bank and capacity edits
    are applied only to :attr:`edited_config`; the source configuration is never
    mutated. Only legs visible under the required carrier filter can be changed.
    A visible leg whose total capacity is zero is retained as an undoable editor
    state, then removed from the edited configuration by :meth:`finish`.

    Parameters
    ----------
    cfg : Config
        Source configuration. It is retained only as the source for a deep copy.
    hub : str
        Airport code whose inbound and outbound legs should be displayed.
    carrier : str
        Required carrier filter. Hidden legs, including legs from other
        carriers with the same bank tag, remain unchanged.
    coloring : str, optional
        Controls leg-line colors. Use ``"carrier"`` (the default),
        ``"spoke"``, ``"spoke.<tag>"``, or ``"leg.<tag>"``.
    bank_tag : str, default "bank"
        Leg-tag key used to store mutually exclusive bank names.
    width : int, default 900
        Preferred widget width in pixels.
    min_height : int, default 180
        Minimum schedule viewport height in pixels.
    max_height : int, default 600
        Maximum schedule viewport height in pixels.
    row_height : int, default 6
        Vertical spacing per leg in pixels.
    """

    _esm = Path(__file__).with_name("_capacity_editor.js")
    _css = Path(__file__).with_name("_schedule_editor.css")

    _legs_data = List(Dict(), default_value=[]).tag(sync=True)
    _settings = Dict(default_value={}).tag(sync=True)
    _state = Dict(default_value={}).tag(sync=True)
    _status = Dict(default_value={}).tag(sync=True)
    _event = Dict(default_value={}).tag(sync=True)
    _revision = Int(0).tag(sync=True)

    def __init__(
        self,
        cfg: Config,
        hub: str,
        *,
        carrier: str,
        coloring: str | None = None,
        bank_tag: str = "bank",
        width: int = 900,
        min_height: int = 180,
        max_height: int = 600,
        row_height: int = 6,
        **widget_kwargs: Any,
    ) -> None:
        if "inplace" in widget_kwargs:
            raise TypeError("HubCapacityEditor does not support in-place editing")
        if not isinstance(cfg, Config):
            raise TypeError("cfg must be a passengersim.config.Config")
        if hub not in cfg.places:
            raise ValueError(f"No place found for {hub}")
        if not isinstance(carrier, str) or not carrier.strip():
            raise ValueError("carrier is required for the hub capacity editor")
        if not isinstance(bank_tag, str) or not bank_tag.strip():
            raise ValueError("bank_tag must be a nonempty string")
        if width < 480:
            raise ValueError("width must be at least 480 pixels")
        if min_height <= 0 or max_height < min_height:
            raise ValueError("heights must be positive and max_height must be at least min_height")
        if isinstance(row_height, bool) or not isinstance(row_height, int) or not 5 <= row_height <= 24:
            raise ValueError("row_height must be an integer from 5 through 24 pixels")

        coloring_source, coloring_tag, coloring_title = parse_coloring(coloring)
        self._source_config = cfg
        self._edited_config = cfg.model_copy(deep=True)
        self.hub = hub
        self.carrier = carrier.strip()
        self.coloring = coloring
        self.bank_tag = bank_tag.strip()
        self.width = width
        self.min_height = min_height
        self.max_height = max_height
        self.row_height = row_height
        self._coloring_source = coloring_source
        self._coloring_tag = coloring_tag
        self._coloring_title = coloring_title

        self._finished = False
        self._leg_by_id = {leg.leg_id: leg for leg in self._edited_config.legs}
        self._visible_leg_ids = self._find_visible_leg_ids()
        self._original_states = self._capture_states()
        self._undo_stack: list[_CapacityEdit] = []
        self._redo_stack: list[_CapacityEdit] = []
        self._active_bank: str | None = None
        self._selected_leg_id: int | None = None
        self._membership_editing = False
        self._staged_leg_ids: set[int] = set()
        self._color_by_group: dict[str, str] = {}
        self._common_capacities = sorted(
            {
                leg.total_capacity
                for leg in self._edited_config.legs
                if leg.carrier == self.carrier and leg.total_capacity > 0
            }
        )
        self._domain_start, self._domain_end = self._initial_time_domain()

        super().__init__(**widget_kwargs)
        self._refresh_widget(message="Select a leg to activate its bank and edit its capacity.")

    @property
    def source_config(self) -> Config:
        """Configuration supplied when the editor was created."""
        return self._source_config

    @property
    def edited_config(self) -> Config:
        """Deep-copied configuration containing bank and capacity edits."""
        return self._edited_config

    @property
    def finished(self) -> bool:
        """Whether editing has been finalized."""
        return self._finished

    @property
    def active_bank(self) -> str | None:
        """Currently highlighted bank name, if any."""
        return self._active_bank

    @property
    def selected_leg_id(self) -> int | None:
        """Currently selected visible leg identifier, if any."""
        return self._selected_leg_id

    @property
    def common_capacities(self) -> tuple[int, ...]:
        """Frozen, ascending carrier capacity presets discovered at initialization."""
        return tuple(self._common_capacities)

    @property
    def banks(self) -> tuple[str, ...]:
        """Sorted bank names currently assigned to visible legs."""
        return tuple(
            sorted(
                {bank for leg_id in self._visible_leg_ids if (bank := self._leg_by_id[leg_id].tags.get(self.bank_tag))}
            )
        )

    @property
    def active_bank_totals(self) -> dict[str, int]:
        """Return directional leg counts and capacity totals for the active bank."""
        return self._totals_for_leg_ids(self._active_member_ids())

    @property
    def changes(self) -> pd.DataFrame:
        """Return current visible-leg bank and capacity changes.

        Returns
        -------
        pandas.DataFrame
            One row per changed visible leg, ordered as displayed.
        """
        rows: list[dict[str, Any]] = []
        for leg_id in self._visible_leg_ids:
            leg = self._leg_by_id[leg_id]
            original = self._original_states[leg_id]
            current_bank = leg.tags.get(self.bank_tag)
            if leg.capacity == original.capacity and current_bank == original.bank:
                continue
            rows.append(
                {
                    "leg_id": leg_id,
                    "carrier": leg.carrier,
                    "fltno": leg.fltno,
                    "orig": leg.orig,
                    "dest": leg.dest,
                    "original_bank": original.bank,
                    "edited_bank": current_bank,
                    "original_capacity": deepcopy(original.capacity),
                    "edited_capacity": deepcopy(leg.capacity),
                    "original_total_capacity": (
                        sum(original.capacity.values()) if isinstance(original.capacity, dict) else original.capacity
                    ),
                    "edited_total_capacity": leg.total_capacity,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "leg_id",
                "carrier",
                "fltno",
                "orig",
                "dest",
                "original_bank",
                "edited_bank",
                "original_capacity",
                "edited_capacity",
                "original_total_capacity",
                "edited_total_capacity",
            ],
        )

    def select_leg(self, leg_id: int) -> None:
        """Select one visible leg and activate its assigned bank."""
        self._require_open()
        self._require_visible_leg(leg_id)
        if self._membership_editing:
            self.toggle_bank_leg(leg_id)
            return
        self._selected_leg_id = leg_id
        self._active_bank = self._leg_by_id[leg_id].tags.get(self.bank_tag)
        self._refresh_widget(message=f"Selected leg {leg_id}.")

    def activate_bank(self, bank_name: str | None) -> None:
        """Activate an existing visible bank, or clear the active bank."""
        self._require_open()
        if self._membership_editing:
            raise RuntimeError("save or cancel the current bank membership edit first")
        if bank_name in (None, ""):
            self._active_bank = None
        else:
            bank_name = _normalize_bank_name(bank_name)
            if bank_name not in self.banks:
                raise ValueError(f"unknown bank {bank_name!r}")
            self._active_bank = bank_name
        self._refresh_widget(message="Active bank selection updated.")

    def begin_bank_edit(self) -> None:
        """Stage edits to the current visible membership of the active bank."""
        self._require_open()
        if self._active_bank is None:
            raise RuntimeError("select or create a bank before editing membership")
        self._membership_editing = True
        self._staged_leg_ids = self._member_ids(self._active_bank)
        self._refresh_widget(message=f"Editing membership for bank {self._active_bank!r}.")

    def new_bank(self, bank_name: str) -> None:
        """Create a staged, initially empty bank."""
        self._require_open()
        if self._membership_editing:
            raise RuntimeError("save or cancel the current bank membership edit first")
        bank_name = _normalize_bank_name(bank_name)
        if bank_name in self.banks:
            raise ValueError(f"bank {bank_name!r} already exists")
        self._active_bank = bank_name
        self._membership_editing = True
        self._staged_leg_ids = set()
        self._refresh_widget(message=f"Select legs for new bank {bank_name!r}, then save.")

    def toggle_bank_leg(self, leg_id: int) -> None:
        """Toggle one visible leg in the staged active-bank membership."""
        self._require_open()
        self._require_visible_leg(leg_id)
        if not self._membership_editing or self._active_bank is None:
            raise RuntimeError("bank membership editing is not active")
        self._selected_leg_id = leg_id
        if leg_id in self._staged_leg_ids:
            self._staged_leg_ids.remove(leg_id)
        else:
            self._staged_leg_ids.add(leg_id)
        self._refresh_widget(message=f"Staged {len(self._staged_leg_ids)} legs for {self._active_bank!r}.")

    def save_bank(self) -> None:
        """Commit the staged active-bank membership to visible leg tags."""
        self._require_open()
        if not self._membership_editing or self._active_bank is None:
            raise RuntimeError("bank membership editing is not active")
        bank_name = self._active_bank
        before = self._capture_states()
        for leg_id in self._member_ids(bank_name) - self._staged_leg_ids:
            self._leg_by_id[leg_id].tags.pop(self.bank_tag, None)
        for leg_id in self._staged_leg_ids:
            self._leg_by_id[leg_id].tags[self.bank_tag] = bank_name
        self._membership_editing = False
        self._staged_leg_ids.clear()
        self._record_edit(before, f"Saved bank {bank_name!r}")

    def cancel_bank_edit(self) -> None:
        """Discard staged bank membership changes."""
        self._require_open()
        if not self._membership_editing:
            return
        was_new = self._active_bank not in self.banks
        self._membership_editing = False
        self._staged_leg_ids.clear()
        if was_new:
            self._active_bank = None
        self._refresh_widget(message="Canceled staged bank membership changes.")

    def rename_bank(self, new_name: str) -> None:
        """Rename the active bank on visible legs only."""
        self._require_open()
        if self._membership_editing:
            raise RuntimeError("save or cancel the current bank membership edit first")
        if self._active_bank is None:
            raise RuntimeError("no bank is active")
        new_name = _normalize_bank_name(new_name)
        old_name = self._active_bank
        if new_name != old_name and new_name in self.banks:
            raise ValueError(f"bank {new_name!r} already exists")
        before = self._capture_states()
        for leg_id in self._member_ids(old_name):
            self._leg_by_id[leg_id].tags[self.bank_tag] = new_name
        self._active_bank = new_name
        self._record_edit(before, f"Renamed bank {old_name!r} to {new_name!r}")

    def delete_bank(self) -> None:
        """Remove the active bank tag from all of its visible legs."""
        self._require_open()
        if self._membership_editing:
            raise RuntimeError("save or cancel the current bank membership edit first")
        if self._active_bank is None:
            raise RuntimeError("no bank is active")
        bank_name = self._active_bank
        before = self._capture_states()
        for leg_id in self._member_ids(bank_name):
            self._leg_by_id[leg_id].tags.pop(self.bank_tag, None)
        self._active_bank = None
        self._record_edit(before, f"Deleted bank {bank_name!r}")

    def set_capacity(self, leg_id: int, capacity: CapacityValue) -> None:
        """Set scalar capacity or per-cabin capacities on one visible leg.

        Parameters
        ----------
        leg_id : int
            Identifier of a visible leg.
        capacity : int or dict[str, int]
            Scalar legs accept a nonnegative integer. Multi-cabin legs require
            a dictionary with exactly their existing cabin keys. A zero scalar
            capacity or an all-zero cabin dictionary marks the leg for removal
            when editing is finalized.
        """
        self._require_open()
        self._require_visible_leg(leg_id)
        leg = self._leg_by_id[leg_id]
        if isinstance(leg.capacity, dict):
            if not isinstance(capacity, dict):
                raise TypeError("multi-cabin legs require a capacity dictionary")
            if set(capacity) != set(leg.capacity):
                raise ValueError("capacity dictionary must preserve the leg's cabin keys")
            validated: CapacityValue = {
                cabin: _validate_capacity_integer(value, label=f"capacity for cabin {cabin}")
                for cabin, value in capacity.items()
            }
        else:
            if isinstance(capacity, dict):
                raise TypeError("single-cabin legs require an integer capacity")
            validated = _validate_capacity_integer(capacity)
        before = self._capture_states()
        leg.capacity = deepcopy(validated)
        self._selected_leg_id = leg_id
        if leg.total_capacity == 0:
            label = f"Marked leg {leg_id} for deletion"
        else:
            label = f"Changed capacity on leg {leg_id}"
        self._record_edit(before, label)

    def undo(self) -> bool:
        """Undo the most recent committed bank or capacity operation."""
        self._require_open()
        if not self._undo_stack:
            return False
        edit = self._undo_stack.pop()
        self._restore_states(edit.before)
        self._redo_stack.append(edit)
        self._cancel_staging_for_history()
        self._refresh_widget(message=f"Undid: {edit.label}.")
        return True

    def redo(self) -> bool:
        """Reapply the most recently undone operation."""
        self._require_open()
        if not self._redo_stack:
            return False
        edit = self._redo_stack.pop()
        self._restore_states(edit.after)
        self._undo_stack.append(edit)
        self._cancel_staging_for_history()
        self._refresh_widget(message=f"Reapplied: {edit.label}.")
        return True

    def reset(self) -> None:
        """Restore original banks and capacities on every visible leg."""
        self._require_open()
        self._restore_states(self._original_states)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._active_bank = None
        self._selected_leg_id = None
        self._membership_editing = False
        self._staged_leg_ids.clear()
        self._refresh_widget(message="Restored original visible-leg banks and capacities.")

    def finish(self) -> Config:
        """Finalize editing and return the independently edited configuration.

        Visible legs with zero total capacity are removed from the edited copy
        during finalization. Hidden legs remain untouched, including hidden legs
        that already have zero capacity.

        Returns
        -------
        Config
            The independently edited configuration without visible
            zero-capacity legs.
        """
        self._require_open()
        if self._membership_editing:
            raise RuntimeError("save or cancel the current bank membership edit before finishing")

        removed_leg_ids = {leg_id for leg_id in self._visible_leg_ids if self._leg_by_id[leg_id].total_capacity == 0}
        if removed_leg_ids:
            # Slice assignment preserves Config's filterable-list wrapper while
            # removing only legs inside this editor's explicit visible scope.
            self._edited_config.legs[:] = [leg for leg in self._edited_config.legs if leg.leg_id not in removed_leg_ids]

        self._finished = True
        self._undo_stack.clear()
        self._redo_stack.clear()
        if removed_leg_ids:
            count = len(removed_leg_ids)
            noun = "leg" if count == 1 else "legs"
            message = f"Capacity and bank editing finished; removed {count} zero-capacity {noun}."
        else:
            message = "Capacity and bank editing finished."
        self._refresh_widget(message=message, phase="finished")
        return self._edited_config

    def _require_open(self) -> None:
        """Reject changes after finalization."""
        if self._finished:
            raise RuntimeError("capacity editing has already been finalized")

    def _require_visible_leg(self, leg_id: int) -> None:
        """Validate that a leg is inside this editor's carrier-filtered scope."""
        if leg_id not in self._visible_leg_ids:
            raise KeyError(f"leg {leg_id} is not visible in this hub capacity editor")

    def _find_visible_leg_ids(self) -> list[int]:
        """Find carrier-filtered hub legs in stable hub-time order."""
        visible = []
        for leg in self._edited_config.legs:
            if leg.carrier != self.carrier:
                continue
            if leg.orig != self.hub and leg.dest != self.hub:
                continue
            hub_time = leg.dep_time if leg.orig == self.hub else leg.arr_time
            visible.append((hub_time, leg.orig, leg.dest, leg.leg_id))
        return [leg_id for _, _, _, leg_id in sorted(visible)]

    def _capture_states(self) -> dict[int, _LegState]:
        """Capture editable state for all visible legs."""
        return {
            leg_id: _LegState(
                capacity=deepcopy(self._leg_by_id[leg_id].capacity),
                bank=self._leg_by_id[leg_id].tags.get(self.bank_tag),
            )
            for leg_id in self._visible_leg_ids
        }

    def _restore_states(self, states: dict[int, _LegState]) -> None:
        """Restore a captured visible-leg state."""
        for leg_id, state in states.items():
            leg = self._leg_by_id[leg_id]
            leg.capacity = deepcopy(state.capacity)
            if state.bank is None:
                leg.tags.pop(self.bank_tag, None)
            else:
                leg.tags[self.bank_tag] = state.bank

    def _record_edit(self, before: dict[int, _LegState], label: str) -> None:
        """Record a committed operation and refresh the widget."""
        after = self._capture_states()
        if after != before:
            self._undo_stack.append(_CapacityEdit(label=label, before=before, after=after))
            self._redo_stack.clear()
        self._refresh_widget(message=f"{label}.")

    def _cancel_staging_for_history(self) -> None:
        """Discard transient membership state before applying history."""
        self._membership_editing = False
        self._staged_leg_ids.clear()

    def _member_ids(self, bank_name: str) -> set[int]:
        """Return visible legs whose configured bank tag matches a name."""
        return {
            leg_id for leg_id in self._visible_leg_ids if self._leg_by_id[leg_id].tags.get(self.bank_tag) == bank_name
        }

    def _active_member_ids(self) -> set[int]:
        """Return committed or staged visible membership for the active bank."""
        if self._active_bank is None:
            return set()
        if self._membership_editing:
            return set(self._staged_leg_ids)
        return self._member_ids(self._active_bank)

    def _totals_for_leg_ids(self, leg_ids: set[int]) -> dict[str, int]:
        """Aggregate arrival and departure counts and total seats."""
        arrivals = [self._leg_by_id[leg_id] for leg_id in leg_ids if self._leg_by_id[leg_id].dest == self.hub]
        departures = [self._leg_by_id[leg_id] for leg_id in leg_ids if self._leg_by_id[leg_id].orig == self.hub]
        return {
            "arrival_legs": len(arrivals),
            "arrival_capacity": sum(leg.total_capacity for leg in arrivals),
            "departure_legs": len(departures),
            "departure_capacity": sum(leg.total_capacity for leg in departures),
        }

    def _initial_time_domain(self) -> tuple[int, int]:
        """Create the same stable, padded domain used by the schedule editor."""
        timestamps = []
        for leg_id in self._visible_leg_ids:
            leg = self._leg_by_id[leg_id]
            timestamps.extend((leg.dep_time, leg.arr_time))
        if timestamps:
            low, high = min(timestamps), max(timestamps)
        else:
            reference = self._edited_config.simulation_controls.reference_epoch()
            low, high = reference - 6 * 3600, reference + 18 * 3600
        midpoint = (low + high) // 2
        half_span = max((high - low) // 2 + 2 * 3600, 15 * 3600)
        return midpoint - half_span, midpoint + half_span

    def _make_leg_payload(self, color_by_leg: dict[int, str]) -> list[dict[str, Any]]:
        """Serialize visible bank and capacity state for the browser view."""
        active_members = self._active_member_ids()
        result = []
        for leg_id in self._visible_leg_ids:
            leg = self._leg_by_id[leg_id]
            outbound = leg.orig == self.hub
            original = self._original_states[leg_id]
            result.append(
                {
                    "leg_id": leg_id,
                    "carrier": leg.carrier,
                    "fltno": leg.fltno,
                    "orig": leg.orig,
                    "dest": leg.dest,
                    "spoke": leg.dest if outbound else leg.orig,
                    "direction": "outbound" if outbound else "inbound",
                    "hub_time": leg.dep_time if outbound else leg.arr_time,
                    "far_time": leg.arr_time if outbound else leg.dep_time,
                    "dep_local": leg.dep_localtime.isoformat(),
                    "arr_local": leg.arr_localtime.isoformat(),
                    "duration_minutes": leg.duration_minutes,
                    "capacity": deepcopy(leg.capacity),
                    "capacity_kind": "cabins" if isinstance(leg.capacity, dict) else "scalar",
                    "total_capacity": leg.total_capacity,
                    "pending_deletion": leg.total_capacity == 0,
                    "bank": leg.tags.get(self.bank_tag),
                    "in_active_bank": leg_id in active_members,
                    "staged": self._membership_editing and leg_id in self._staged_leg_ids,
                    "selected": leg_id == self._selected_leg_id,
                    "color_group": leg_color_group(
                        self._edited_config,
                        self.hub,
                        leg,
                        self._coloring_source,
                        self._coloring_tag,
                    ),
                    "color": color_by_leg[leg_id],
                    "changed": (leg.capacity != original.capacity or leg.tags.get(self.bank_tag) != original.bank),
                }
            )
        return result

    def _refresh_widget(self, *, message: str, phase: str | None = None) -> None:
        """Synchronize Python state to the widget front end."""
        if not self._membership_editing and self._active_bank not in self.banks:
            self._active_bank = None
        visible_legs = [self._leg_by_id[leg_id] for leg_id in self._visible_leg_ids]
        color_by_leg, legend, self._color_by_group = color_scale(
            self._edited_config,
            self.hub,
            visible_legs,
            self._coloring_source,
            self._coloring_tag,
            existing_colors=self._color_by_group,
        )
        self._legs_data = self._make_leg_payload(color_by_leg)
        self._settings = {
            "hub": self.hub,
            "hub_timezone": self._edited_config.places[self.hub].time_zone,
            "carrier": self.carrier,
            "coloring": self.coloring or "carrier",
            "coloring_title": self._coloring_title,
            "legend": legend,
            "bank_tag": self.bank_tag,
            "common_capacities": self._common_capacities,
            "width": self.width,
            "min_height": self.min_height,
            "max_height": self.max_height,
            "row_height": self.row_height,
            "domain_start": self._domain_start,
            "domain_end": self._domain_end,
        }
        self._state = {
            "banks": list(self.banks),
            "active_bank": self._active_bank,
            "selected_leg_id": self._selected_leg_id,
            "membership_editing": self._membership_editing,
            "staged_leg_ids": sorted(self._staged_leg_ids),
            "totals": self.active_bank_totals,
        }
        self._status = {
            "phase": phase or ("finished" if self._finished else "editing"),
            "message": message,
            "modified_count": len(self.changes),
            "can_undo": bool(self._undo_stack) and not self._finished,
            "can_redo": bool(self._redo_stack) and not self._finished,
            "finished": self._finished,
        }
        self._revision += 1

    @observe("_event")
    def _handle_frontend_event(self, change: dict[str, Any]) -> None:
        """Dispatch browser events to testable Python editing methods."""
        event = change["new"]
        event_type = event.get("type")
        try:
            if event_type == "select_leg":
                self.select_leg(int(event["leg_id"]))
            elif event_type == "activate_bank":
                self.activate_bank(event.get("bank"))
            elif event_type == "begin_bank_edit":
                self.begin_bank_edit()
            elif event_type == "new_bank":
                self.new_bank(event.get("bank"))
            elif event_type == "toggle_bank_leg":
                self.toggle_bank_leg(int(event["leg_id"]))
            elif event_type == "save_bank":
                self.save_bank()
            elif event_type == "cancel_bank_edit":
                self.cancel_bank_edit()
            elif event_type == "rename_bank":
                self.rename_bank(event.get("bank"))
            elif event_type == "delete_bank":
                self.delete_bank()
            elif event_type == "set_capacity":
                self.set_capacity(int(event["leg_id"]), event.get("capacity"))
            elif event_type == "undo":
                self.undo()
            elif event_type == "redo":
                self.redo()
            elif event_type == "reset":
                self.reset()
            elif event_type == "finish":
                self.finish()
            elif event_type is not None:
                raise ValueError(f"unknown hub capacity editor event {event_type!r}")
        except Exception as err:  # UI errors should remain visible without losing the widget
            self._status = {
                **self._status,
                "phase": "error",
                "message": f"{type(err).__name__}: {err}",
            }
            self._revision += 1


def edit_hub_capacity(
    cfg: Config,
    hub: str,
    *,
    carrier: str,
    coloring: str | None = None,
    bank_tag: str = "bank",
    width: int = 900,
    min_height: int = 180,
    max_height: int = 600,
    row_height: int = 6,
) -> HubCapacityEditor:
    """Create a copy-only interactive hub bank and capacity editor.

    Setting a visible leg's total capacity to zero marks it for deletion. The
    leg remains available to undo or edit until :meth:`HubCapacityEditor.finish`
    removes it from the returned configuration copy.

    Parameters
    ----------
    cfg : Config
        Source configuration. The editor immediately makes a deep copy.
    hub : str
        Airport code whose schedule should be edited.
    carrier : str
        Required carrier whose visible hub legs may be edited.
    coloring : str, optional
        Color by carrier, spoke, a spoke-place tag using ``"spoke.<tag>"``,
        or a leg tag using ``"leg.<tag>"``.
    bank_tag : str, default "bank"
        Leg-tag key used to store bank names.
    width : int, default 900
        Preferred widget width in pixels.
    min_height : int, default 180
        Minimum schedule viewport height in pixels.
    max_height : int, default 600
        Maximum schedule viewport height in pixels.
    row_height : int, default 6
        Vertical spacing per leg in pixels.

    Returns
    -------
    HubCapacityEditor
        Displayable Jupyter widget containing an independently edited Config.
    """
    return HubCapacityEditor(
        cfg,
        hub,
        carrier=carrier,
        coloring=coloring,
        bank_tag=bank_tag,
        width=width,
        min_height=min_height,
        max_height=max_height,
        row_height=row_height,
    )
