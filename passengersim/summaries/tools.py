from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from . import SimulationTables


def aggregate_by_concat_dataframe(
    name: str,
) -> Callable:
    """Create function to aggregate from summaries by simple concatenation."""

    def concat_dataframe(
        summaries: list[SimulationTables],
    ) -> pd.DataFrame | None:
        frames = []
        for s in summaries:
            frame = getattr(s, f"_raw_{name}", None)
            if frame is not None:
                frames.append(frame)
        if frames:
            return pd.concat(frames)
        return None

    return concat_dataframe


def aggregate_by_summing_dataframe(
    name: str,
    extra_idxs: list[str] = None,
) -> Callable[[list[SimulationTables]], pd.DataFrame | None]:
    """Create function to aggregate from summaries by summing."""

    def sum_dataframe(
        summaries: list[SimulationTables],
    ) -> pd.DataFrame | None:
        frames = []
        for s in summaries:
            frame = getattr(s, f"_raw_{name}", None)
            if frame is not None:
                if extra_idxs:
                    frame = frame.set_index(extra_idxs, append=True)
                frames.append(frame)
        while len(frames) > 1:
            frames[0] = frames[0].add(frames.pop(1), fill_value=0)
        if frames:
            return frames[0].reset_index(extra_idxs)
        return None

    return sum_dataframe
