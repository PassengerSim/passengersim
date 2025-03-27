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
    ) -> pd.DataFrame | None | Exception:
        frames = []
        for s in summaries:
            frame = getattr(s, f"_raw_{name}", None)
            if isinstance(frame, Exception):
                return frame
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
    ) -> pd.DataFrame | None | Exception:
        frames = []
        for s in summaries:
            frame = getattr(s, f"_raw_{name}", None)
            if isinstance(frame, Exception):
                return frame
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
    ) -> pd.DataFrame | None | Exception:
        frames = []
        for s in summaries:
            frame = getattr(s, f"_raw_{name}", None)
            if isinstance(frame, Exception):
                return frame
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


def aggregate_by_first_dataframe(
    name: str,
) -> Callable[[list[SimulationTables]], pd.DataFrame | None]:
    """Create function to aggregate from summaries by taking the first."""

    def first_dataframe(
        summaries: list[SimulationTables],
    ) -> pd.DataFrame | None | Exception:
        for s in summaries:
            frame = getattr(s, f"_raw_{name}", None)
            if frame is not None:
                return frame
        return None

    return first_dataframe


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


def break_on_integer(
    s: pd.Series, breakpoints: tuple[int, ...], minimum=0, maximum=100, result_name=None
):
    """Break a series into categories based on integer breakpoints.

    Parameters
    ----------
    s : pd.Series
        The series to break into categories.
    breakpoints : tuple of int
        The breakpoints for the categories. If the first breakpoint is less than
        the minimum value it is moved up, and if the last breakpoint is greater
        than the maximum value it is moved down.
    minimum : int, default 0
        The minimum value for the series. Values less than this will be assigned
        to the first category.
    maximum : int, default 100
        The maximum value for the series. Values greater than this will be
        assigned to the last category.
    result_name : str, default None
        The name to assign to the resulting series.

    Returns
    -------
    pd.Series
        A series with categorical values.
    """
    if not isinstance(breakpoints, tuple):
        breakpoints = tuple(breakpoints)
    if breakpoints[0] <= minimum:
        breakpoints = (minimum - 1,) + breakpoints[1:]
    else:
        breakpoints = (minimum - 1,) + breakpoints
    if breakpoints[-1] >= maximum + 1:
        breakpoints = breakpoints[:-1] + (maximum + 1,)
    else:
        breakpoints = breakpoints + (maximum + 1,)

    # Create labels for categories
    def make_label(i, j):
        if i == j - 1:
            return f"{i}"
        else:
            return f"{i}-{j-1}"

    labels = [make_label(0, breakpoints[1])]
    for i in range(1, len(breakpoints) - 2):
        labels += [make_label(breakpoints[i], breakpoints[i + 1])]
    if breakpoints[-2] < 100:
        labels += [make_label(breakpoints[-2], maximum + 1)]
    else:
        labels += [str(maximum)]
    breaker = pd.cut(
        s,
        bins=breakpoints,
        right=False,
        labels=labels,
    )
    if result_name:
        breaker = breaker.rename(result_name)
    return breaker
