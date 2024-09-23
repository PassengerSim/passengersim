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
            if extra_idxs:
                return frames[0].reset_index(extra_idxs)
            return frames[0]
        return None

    return sum_dataframe


def aggregate_by_averaging_dataframe(
    name: str,
    extra_idxs: list[str] = None,
) -> Callable[[list[SimulationTables]], pd.DataFrame | None]:
    """Create function to aggregate from summaries by summing."""

    def avg_dataframe(
        summaries: list[SimulationTables],
    ) -> pd.DataFrame | None:
        frames = []
        for s in summaries:
            frame = getattr(s, f"_raw_{name}", None)
            if frame is not None:
                if extra_idxs:
                    frame = frame.set_index(extra_idxs, append=True)
                frames.append(frame)
        n = len(frames)
        while len(frames) > 1:
            frames[0] = frames[0].add(frames.pop(1), fill_value=0)
        if frames:
            result = frames[0] / n
            if extra_idxs:
                result = result.reset_index(extra_idxs)
            return result
        return None

    return avg_dataframe


def total_sum_of_squares(mu, sigma, n):
    return (mu**2 + (sigma**2) * ((n - 1) / (n))) * (n)


def total_sum(mu, n):
    return mu * n


def combine_sigmas(sigma, sigma2, mu, mu2, n, n2, ddof=0):
    nn = n + n2
    mean_sq = (
        total_sum_of_squares(mu, sigma, n) + total_sum_of_squares(mu2, sigma2, n2)
    ) / nn
    sq_mean = ((total_sum(mu, n) + total_sum(mu2, n2)) / (nn)) ** 2
    adj = nn / (nn - ddof)
    raw = mean_sq - sq_mean
    return raw * adj
