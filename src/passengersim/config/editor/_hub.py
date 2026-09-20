"""Shared helpers for hub-centered notebook editors."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from passengersim.config.base import Config
from passengersim.config.legs import Leg
from passengersim.utils.colors import common_carrier_colors, passengersim_colors

MISSING_TAG = "(missing)"


def parse_coloring(coloring: str | None) -> tuple[str, str | None, str]:
    """Normalize a coloring selector into its source, tag, and legend title."""
    if coloring is None or coloring == "carrier":
        return "carrier", None, "Carrier"
    if not isinstance(coloring, str):
        raise TypeError("coloring must be a string or None")
    if coloring == "spoke":
        return "spoke", None, "Spoke"
    if coloring.startswith("spoke."):
        tag = coloring.removeprefix("spoke.")
        if tag:
            return "spoke_tag", tag, f"Spoke {tag}"
    if coloring.startswith("leg."):
        tag = coloring.removeprefix("leg.")
        if tag:
            return "leg_tag", tag, f"Leg {tag}"
    raise ValueError("unrecognized coloring; use None, 'carrier', 'spoke', 'spoke.<tag>', or 'leg.<tag>'")


def leg_color_group(
    cfg: Config,
    hub: str,
    leg: Leg,
    coloring_source: str,
    coloring_tag: str | None,
) -> str:
    """Return one leg's categorical value for a normalized coloring selector."""
    spoke = leg.dest if leg.orig == hub else leg.orig
    if coloring_source == "carrier":
        return leg.carrier
    if coloring_source == "spoke":
        return spoke
    if coloring_source == "spoke_tag":
        value = cfg.places[spoke].tags.get(coloring_tag)
    else:
        value = leg.tags.get(coloring_tag)
    return str(value) if value not in (None, "") else MISSING_TAG


def color_scale(
    cfg: Config,
    hub: str,
    legs: Iterable[Leg],
    coloring_source: str,
    coloring_tag: str | None,
    *,
    existing_colors: dict[str, str] | None = None,
) -> tuple[dict[int, str], list[dict[str, Any]], dict[str, str]]:
    """Build deterministic colors and legend entries for visible legs.

    Existing category assignments are retained when provided, allowing editors
    that modify tags to keep colors stable throughout a session.
    """
    legs = list(legs)
    group_by_leg = {leg.leg_id: leg_color_group(cfg, hub, leg, coloring_source, coloring_tag) for leg in legs}
    groups = sorted(set(group_by_leg.values()))
    color_by_group = dict(existing_colors or {})

    if coloring_source == "carrier":
        for group in groups:
            if group in color_by_group:
                continue
            try:
                color_by_group[group] = common_carrier_colors(group, error_if_missing=True)
            except KeyError:
                pass

    used_colors = set(color_by_group.values())
    available = [color for color in passengersim_colors if color not in used_colors]
    recycled_index = 0
    for group in groups:
        if group in color_by_group:
            continue
        if available:
            color = available.pop(0)
        else:
            color = passengersim_colors[recycled_index % len(passengersim_colors)]
            recycled_index += 1
        color_by_group[group] = color
        used_colors.add(color)

    return (
        {leg_id: color_by_group[group] for leg_id, group in group_by_leg.items()},
        [{"label": group, "color": color_by_group[group]} for group in groups],
        color_by_group,
    )
