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
