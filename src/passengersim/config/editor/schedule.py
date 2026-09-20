"""Interactive tools for editing flight-leg schedules in Jupyter notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from traitlets import Dict, Int, List, observe

try:
    import anywidget
except ImportError as err:  # pragma: no cover - exercised in installations without the lab extra
    raise ImportError(
        "The schedule editor requires Jupyter widget dependencies. Install PassengerSim with the 'lab' extra."
    ) from err

from passengersim.config.base import Config
from passengersim.config.legs import Leg

from ._hub import color_scale, leg_color_group, parse_coloring

__all__ = ["ScheduleEditor", "edit_hub_schedule"]


@dataclass(frozen=True)
class _LegTiming:
    """All timing fields changed as one atomic schedule edit."""

    dep_time: int
    arr_time: int
    date: datetime
    arr_day: int
    dep_time_offset: int
    arr_time_offset: int
    time_adjusted: bool


@dataclass(frozen=True)
class _Edit:
    """A reversible change to one leg."""

    leg_id: int
    before: _LegTiming
    after: _LegTiming


def _capture_timing(leg: Leg) -> _LegTiming:
    """Capture the timing state of a leg."""
    return _LegTiming(
        dep_time=leg.dep_time,
        arr_time=leg.arr_time,
        date=leg.date,
        arr_day=leg.arr_day,
        dep_time_offset=leg.dep_time_offset,
        arr_time_offset=leg.arr_time_offset,
        time_adjusted=leg.time_adjusted,
    )


def _restore_timing(leg: Leg, timing: _LegTiming) -> None:
    """Restore a previously captured timing state without revalidating the config."""
    leg.dep_time = timing.dep_time
    leg.arr_time = timing.arr_time
    leg.date = timing.date
    leg.arr_day = timing.arr_day
    leg.dep_time_offset = timing.dep_time_offset
    leg.arr_time_offset = timing.arr_time_offset
    leg.time_adjusted = timing.time_adjusted


def _local_datetime(timestamp: int, timezone: str | None) -> datetime:
    """Convert a Unix timestamp to a leg endpoint's local datetime."""
    result = datetime.fromtimestamp(timestamp, tz=UTC)
    if timezone is not None:
        result = result.astimezone(ZoneInfo(timezone))
    return result


def _shift_leg(leg: Leg, delta_seconds: int) -> None:
    """Shift one leg while preserving elapsed duration and local date metadata."""
    if isinstance(delta_seconds, bool) or not isinstance(delta_seconds, int):
        raise TypeError("delta_seconds must be an integer")

    duration = leg.arr_time - leg.dep_time
    if duration <= 0:
        raise ValueError(f"leg {leg.leg_id} has a non-positive duration")

    dep_time = leg.dep_time + delta_seconds
    arr_time = leg.arr_time + delta_seconds
    dep_local = _local_datetime(dep_time, leg.orig_timezone)
    arr_local = _local_datetime(arr_time, leg.dest_timezone)

    # Leg.date represents the origin-local calendar date, but uses UTC as a
    # neutral timezone in parsed configurations. Human-readable serialization
    # derives the same date from dep_localtime.
    leg.date = datetime(dep_local.year, dep_local.month, dep_local.day, tzinfo=UTC)
    leg.dep_time = dep_time
    leg.arr_time = arr_time
    leg.arr_day = (arr_local.date() - dep_local.date()).days
    leg.dep_time_offset = int((dep_local.utcoffset() or timedelta()).total_seconds())
    leg.arr_time_offset = int((arr_local.utcoffset() or timedelta()).total_seconds())
    leg.time_adjusted = True

    if leg.arr_time - leg.dep_time != duration:  # pragma: no cover - defensive invariant
        raise AssertionError("schedule edit changed the leg duration")


_parse_coloring = parse_coloring


class ScheduleEditor(anywidget.AnyWidget):
    """Edit the schedule of flight legs touching one hub.

    The supplied configuration is always deep-copied. All edits, including the
    final connection rebuild, are applied only to :attr:`edited_config`; the
    original configuration is never mutated.

    Parameters
    ----------
    cfg : Config
        Source configuration. It is retained only as the source for a deep copy.
    hub : str
        Airport code whose inbound and outbound legs should be displayed.
    carrier : str, optional
        If provided, display only legs operated by this carrier.
    coloring : str, optional
        Controls leg-line colors. Use ``"carrier"`` (the default),
        ``"spoke"``, ``"spoke.<tag>"`` for a spoke-place tag, or
        ``"leg.<tag>"`` for a leg tag. Missing tag values form a separate
        ``"(missing)"`` category.
    snap_minutes : int, default 5
        Initial drag snapping interval in minutes. The interval can also be
        changed from the widget toolbar.
    width : int, default 900
        Preferred widget width in pixels.
    min_height : int, default 180
        Minimum schedule viewport height in pixels.
    max_height : int, default 600
        Maximum schedule viewport height in pixels.
    row_height : int, default 6
        Vertical spacing per leg in pixels. The default fits at least 80 legs
        in the default viewport; values from 5 through 24 are accepted.

    Notes
    -----
    Clicking ``Finish & rebuild connections`` replaces connections on the edited
    copy using its revised schedule. Users can explicitly select
    ``Skip connection rebuild`` before finishing.
    """

    _esm = Path(__file__).with_name("_schedule_editor.js")
    _css = Path(__file__).with_name("_schedule_editor.css")

    _legs_data = List(Dict(), default_value=[]).tag(sync=True)
    _settings = Dict(default_value={}).tag(sync=True)
    _status = Dict(default_value={}).tag(sync=True)
    _event = Dict(default_value={}).tag(sync=True)
    _revision = Int(0).tag(sync=True)

    def __init__(
        self,
        cfg: Config,
        hub: str,
        *,
        carrier: str | None = None,
        coloring: str | None = None,
        snap_minutes: int = 5,
        width: int = 900,
        min_height: int = 180,
        max_height: int = 600,
        row_height: int = 6,
        **widget_kwargs: Any,
    ) -> None:
        if "inplace" in widget_kwargs:
            raise TypeError("ScheduleEditor does not support in-place editing")
        if not isinstance(cfg, Config):
            raise TypeError("cfg must be a passengersim.config.Config")
        if hub not in cfg.places:
            raise ValueError(f"No place found for {hub}")
        if isinstance(snap_minutes, bool) or not isinstance(snap_minutes, int) or snap_minutes <= 0:
            raise ValueError("snap_minutes must be a positive integer")
        if width < 480:
            raise ValueError("width must be at least 480 pixels")
        if min_height <= 0 or max_height < min_height:
            raise ValueError("heights must be positive and max_height must be at least min_height")
        if isinstance(row_height, bool) or not isinstance(row_height, int) or not 5 <= row_height <= 24:
            raise ValueError("row_height must be an integer from 5 through 24 pixels")

        coloring_source, coloring_tag, coloring_title = _parse_coloring(coloring)

        self._source_config = cfg
        self._edited_config = cfg.model_copy(deep=True)
        self.hub = hub
        self.carrier = carrier
        self.coloring = coloring
        self._coloring_source = coloring_source
        self._coloring_tag = coloring_tag
        self._coloring_title = coloring_title
        self.snap_minutes = snap_minutes
        self.width = width
        self.min_height = min_height
        self.max_height = max_height
        self.row_height = row_height
        self._finished = False
        self._connections_rebuilt = False
        self._undo_stack: list[_Edit] = []
        self._redo_stack: list[_Edit] = []
        self._leg_by_id = {leg.leg_id: leg for leg in self._edited_config.legs}
        self._original_timing = {leg.leg_id: _capture_timing(leg) for leg in self._edited_config.legs}
        self._visible_leg_ids = self._find_visible_leg_ids()
        self._connections_map = self._build_connections_map()
        self._domain_start, self._domain_end = self._initial_time_domain()

        super().__init__(**widget_kwargs)
        self._refresh_widget(message="Drag a flight left or right to change its schedule.")

    @property
    def source_config(self) -> Config:
        """Config supplied when the editor was created.

        This object is exposed for identity checks and is never mutated by the
        editor.
        """
        return self._source_config

    @property
    def edited_config(self) -> Config:
        """Deep-copied configuration containing the schedule edits."""
        return self._edited_config

    @property
    def finished(self) -> bool:
        """Whether editing has been finalized."""
        return self._finished

    @property
    def connections_rebuilt(self) -> bool:
        """Whether connections were rebuilt when editing was finalized."""
        return self._connections_rebuilt

    @property
    def changes(self) -> pd.DataFrame:
        """Return the current schedule changes as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            One row per modified leg, ordered as displayed in the editor.
        """
        rows = []
        for leg_id in self._visible_leg_ids:
            leg = self._leg_by_id[leg_id]
            original = self._original_timing[leg_id]
            if leg.dep_time == original.dep_time and leg.arr_time == original.arr_time:
                continue
            rows.append(
                {
                    "leg_id": leg_id,
                    "carrier": leg.carrier,
                    "fltno": leg.fltno,
                    "orig": leg.orig,
                    "dest": leg.dest,
                    "original_dep_time": _local_datetime(original.dep_time, leg.orig_timezone),
                    "edited_dep_time": leg.dep_localtime,
                    "original_arr_time": _local_datetime(original.arr_time, leg.dest_timezone),
                    "edited_arr_time": leg.arr_localtime,
                    "delta_minutes": (leg.dep_time - original.dep_time) // 60,
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
                "original_dep_time",
                "edited_dep_time",
                "original_arr_time",
                "edited_arr_time",
                "delta_minutes",
            ],
        )

    def move_leg(self, leg_id: int, delta_minutes: int) -> None:
        """Move one visible leg by a number of minutes.

        Parameters
        ----------
        leg_id : int
            Unique identifier of a leg visible in this editor.
        delta_minutes : int
            Signed number of minutes to move the leg.

        Raises
        ------
        RuntimeError
            If the editor has already been finalized.
        KeyError
            If the leg is not visible in this hub editor.
        TypeError
            If ``delta_minutes`` is not an integer.
        """
        if self._finished:
            raise RuntimeError("schedule editing has already been finalized")
        if leg_id not in self._visible_leg_ids:
            raise KeyError(f"leg {leg_id} is not visible in this schedule editor")
        if isinstance(delta_minutes, bool) or not isinstance(delta_minutes, int):
            raise TypeError("delta_minutes must be an integer")
        if delta_minutes == 0:
            return

        leg = self._leg_by_id[leg_id]
        before = _capture_timing(leg)
        _shift_leg(leg, delta_minutes * 60)
        after = _capture_timing(leg)
        self._undo_stack.append(_Edit(leg_id=leg_id, before=before, after=after))
        self._redo_stack.clear()
        self._refresh_widget(message=f"Moved leg {leg_id} by {delta_minutes:+d} minutes.")

    def undo(self) -> bool:
        """Undo the most recent leg move.

        Returns
        -------
        bool
            True if a move was undone, otherwise False.
        """
        if self._finished:
            raise RuntimeError("schedule editing has already been finalized")
        if not self._undo_stack:
            return False
        edit = self._undo_stack.pop()
        _restore_timing(self._leg_by_id[edit.leg_id], edit.before)
        self._redo_stack.append(edit)
        self._refresh_widget(message=f"Undid the last move to leg {edit.leg_id}.")
        return True

    def redo(self) -> bool:
        """Reapply the most recently undone leg move.

        Returns
        -------
        bool
            True if a move was reapplied, otherwise False.
        """
        if self._finished:
            raise RuntimeError("schedule editing has already been finalized")
        if not self._redo_stack:
            return False
        edit = self._redo_stack.pop()
        _restore_timing(self._leg_by_id[edit.leg_id], edit.after)
        self._undo_stack.append(edit)
        self._refresh_widget(message=f"Reapplied the move to leg {edit.leg_id}.")
        return True

    def reset(self) -> None:
        """Restore every visible leg to its original schedule."""
        if self._finished:
            raise RuntimeError("schedule editing has already been finalized")
        for leg_id in self._visible_leg_ids:
            _restore_timing(self._leg_by_id[leg_id], self._original_timing[leg_id])
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._refresh_widget(message="Restored the original schedule.")

    def finish(self, *, rebuild_connections: bool = True) -> Config:
        """Finalize editing and optionally rebuild connections.

        Parameters
        ----------
        rebuild_connections : bool, default True
            If True, replace all paths using the connection builder and the
            edited schedule. Set this to False to explicitly skip rebuilding.

        Returns
        -------
        Config
            The edited configuration. This is always distinct from the supplied
            source configuration.

        Raises
        ------
        RuntimeError
            If editing was already finalized.
        """
        if self._finished:
            raise RuntimeError("schedule editing has already been finalized")
        if not isinstance(rebuild_connections, bool):
            raise TypeError("rebuild_connections must be a boolean")

        if rebuild_connections:
            self._set_status(
                phase="rebuilding",
                message="Rebuilding connections from the edited schedule…",
            )
            from passengersim.connection_builder import prebuild_connections

            # Replacement is essential: keeping existing paths would preserve the
            # connection set built from the original leg timings.
            try:
                prebuild_connections(self._edited_config, inplace=True, existing_paths="replace")
            except Exception as err:
                self._set_status(phase="error", message=f"Connection rebuild failed: {err}")
                self._revision += 1
                raise
            self._connections_rebuilt = True
            self._connections_map = self._build_connections_map()
            message = "Editing finished and connections were rebuilt."
        else:
            self._connections_rebuilt = False
            message = "Editing finished without rebuilding connections."

        self._finished = True
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._refresh_widget(message=message, phase="finished")
        return self._edited_config

    def _find_visible_leg_ids(self) -> list[int]:
        """Find hub legs once so row order stays stable while dragging."""
        visible = []
        for leg in self._edited_config.legs:
            if leg.orig != self.hub and leg.dest != self.hub:
                continue
            if self.carrier is not None and leg.carrier != self.carrier:
                continue
            hub_time = leg.dep_time if leg.orig == self.hub else leg.arr_time
            visible.append((hub_time, leg.orig, leg.dest, leg.leg_id))
        return [leg_id for _, _, _, leg_id in sorted(visible)]

    def _build_connections_map(self) -> dict[int, set[int]]:
        """Build click-highlighting relationships from the current static paths."""
        visible = set(self._visible_leg_ids)
        result = {leg_id: {leg_id} for leg_id in visible}
        for path in self._edited_config.paths:
            path_legs = visible.intersection(path.legs)
            for leg_id in path_legs:
                result[leg_id].update(path_legs)
        return result

    def _initial_time_domain(self) -> tuple[int, int]:
        """Create a stable, padded time domain that permits cross-midnight moves."""
        timestamps = []
        for leg_id in self._visible_leg_ids:
            leg = self._leg_by_id[leg_id]
            timestamps.extend((leg.dep_time, leg.arr_time))
        if timestamps:
            low, high = min(timestamps), max(timestamps)
        else:
            reference = self._edited_config.simulation_controls.reference_epoch()
            low, high = reference - 6 * 3600, reference + 18 * 3600

        # Guarantee at least 30 hours of visible time, centered on the schedule,
        # and add padding for easier pointer capture at each edge.
        midpoint = (low + high) // 2
        half_span = max((high - low) // 2 + 2 * 3600, 15 * 3600)
        return midpoint - half_span, midpoint + half_span

    def _leg_color_group(self, leg: Leg) -> str:
        """Return the categorical coloring value for one visible leg."""
        return leg_color_group(
            self._edited_config,
            self.hub,
            leg,
            self._coloring_source,
            self._coloring_tag,
        )

    def _color_scale(self) -> tuple[dict[int, str], list[dict[str, str]]]:
        """Build stable colors and compact legend entries for visible legs."""
        color_by_leg, legend, _ = color_scale(
            self._edited_config,
            self.hub,
            (self._leg_by_id[leg_id] for leg_id in self._visible_leg_ids),
            self._coloring_source,
            self._coloring_tag,
        )
        return color_by_leg, legend

    def _make_leg_payload(self) -> list[dict[str, Any]]:
        """Serialize visible leg state for the browser view."""
        result = []
        color_by_leg, _ = self._color_scale()
        for leg_id in self._visible_leg_ids:
            leg = self._leg_by_id[leg_id]
            outbound = leg.orig == self.hub
            original = self._original_timing[leg_id]
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
                    "dep_time": leg.dep_time,
                    "arr_time": leg.arr_time,
                    "dep_local": leg.dep_localtime.isoformat(),
                    "arr_local": leg.arr_localtime.isoformat(),
                    "duration_minutes": leg.duration_minutes,
                    "capacity": leg.total_capacity,
                    "color_group": self._leg_color_group(leg),
                    "color": color_by_leg[leg_id],
                    "changed": leg.dep_time != original.dep_time,
                    "delta_minutes": (leg.dep_time - original.dep_time) // 60,
                    "connected_leg_ids": sorted(self._connections_map.get(leg_id, {leg_id})),
                }
            )
        return result

    def _refresh_widget(self, *, message: str, phase: str | None = None) -> None:
        """Synchronize current Python state to the widget front end."""
        hub_tz = self._edited_config.places[self.hub].time_zone
        _, legend = self._color_scale()
        self._legs_data = self._make_leg_payload()
        self._settings = {
            "hub": self.hub,
            "hub_timezone": hub_tz,
            "carrier": self.carrier,
            "coloring": self.coloring or "carrier",
            "coloring_title": self._coloring_title,
            "legend": legend,
            "snap_minutes": self.snap_minutes,
            "width": self.width,
            "min_height": self.min_height,
            "max_height": self.max_height,
            "row_height": self.row_height,
            "domain_start": self._domain_start,
            "domain_end": self._domain_end,
        }
        self._set_status(phase=phase or ("finished" if self._finished else "editing"), message=message)
        self._revision += 1

    def _set_status(self, *, phase: str, message: str) -> None:
        """Synchronize controls and a user-facing status message."""
        self._status = {
            "phase": phase,
            "message": message,
            "modified_count": len(self.changes),
            "can_undo": bool(self._undo_stack) and not self._finished,
            "can_redo": bool(self._redo_stack) and not self._finished,
            "finished": self._finished,
            "connections_rebuilt": self._connections_rebuilt,
        }

    @observe("_event")
    def _handle_frontend_event(self, change: dict[str, Any]) -> None:
        """Dispatch one browser event to the testable Python editing methods."""
        event = change["new"]
        event_type = event.get("type")
        try:
            if event_type == "move":
                delta_seconds = event.get("delta_seconds")
                if isinstance(delta_seconds, bool) or not isinstance(delta_seconds, int):
                    raise TypeError("delta_seconds must be an integer")
                if delta_seconds % 60:
                    raise ValueError("schedule moves must use whole minutes")
                self.move_leg(int(event["leg_id"]), delta_seconds // 60)
            elif event_type == "undo":
                self.undo()
            elif event_type == "redo":
                self.redo()
            elif event_type == "reset":
                self.reset()
            elif event_type == "set_snap":
                snap_minutes = event.get("snap_minutes")
                if isinstance(snap_minutes, bool) or not isinstance(snap_minutes, int) or snap_minutes <= 0:
                    raise ValueError("snap_minutes must be a positive integer")
                self.snap_minutes = snap_minutes
                self._refresh_widget(message=f"Drag snapping set to {snap_minutes} minutes.")
            elif event_type == "finish":
                self.finish(rebuild_connections=not bool(event.get("skip_rebuild", False)))
            elif event_type is not None:
                raise ValueError(f"unknown schedule editor event {event_type!r}")
        except Exception as err:  # UI errors should be visible without losing the widget
            self._set_status(phase="error", message=f"{type(err).__name__}: {err}")
            self._revision += 1


def edit_hub_schedule(
    cfg: Config,
    hub: str,
    *,
    carrier: str | None = None,
    coloring: str | None = None,
    snap_minutes: int = 5,
    width: int = 900,
    min_height: int = 180,
    max_height: int = 600,
    row_height: int = 6,
) -> ScheduleEditor:
    """Create a copy-only interactive hub schedule editor.

    Parameters
    ----------
    cfg : Config
        Source configuration. The editor immediately makes a deep copy.
    hub : str
        Airport code whose schedule should be edited.
    carrier : str, optional
        If provided, show only this carrier's legs.
    coloring : str, optional
        Color by carrier (the default), spoke, a spoke-place tag with
        ``"spoke.<tag>"``, or a leg tag with ``"leg.<tag>"``.
    snap_minutes : int, default 5
        Initial drag snapping interval in minutes.
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
    ScheduleEditor
        Displayable Jupyter widget containing an independently edited Config.
    """
    return ScheduleEditor(
        cfg,
        hub,
        carrier=carrier,
        coloring=coloring,
        snap_minutes=snap_minutes,
        width=width,
        min_height=min_height,
        max_height=max_height,
        row_height=row_height,
    )
