"""Mergeable native leg bid-price moments, independent of database recording."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import pandas as pd

from .generic import GenericSimulationTables, MissingDataError, SimulationTableItem

if TYPE_CHECKING:
    from passengersim import Simulation


def _moments_from_rows(rows: list[tuple[int, int, int, float, float]]) -> pd.DataFrame:
    return (
        pd.DataFrame(rows, columns=["leg_id", "days_prior", "n_observations", "mean", "M2"])
        .astype(
            {"leg_id": "int64", "days_prior": "int64", "n_observations": "int64", "mean": "float64", "M2": "float64"}
        )
        .set_index(["leg_id", "days_prior"])
        .sort_index()
    )


def _merge_moments(left: pd.DataFrame | None, right: pd.DataFrame | None) -> pd.DataFrame | None:
    """Pool Welford moments by observation count without modifying either input."""
    if left is None:
        return None if right is None else right.copy()
    if right is None:
        return left.copy()
    index = left.index.union(right.index).sort_values()
    a, b = left.reindex(index), right.reindex(index)
    na = a.n_observations.fillna(0).astype("int64")
    nb = b.n_observations.fillna(0).astype("int64")
    n = na + nb
    # Unobserved/missing cells are identities, not NaN observations. Retain NaNs
    # in genuinely observed means rather than silently treating those as zeros.
    ma, mb = a["mean"].where(na > 0, 0.0), b["mean"].where(nb > 0, 0.0)
    delta = mb - ma
    weight = nb / n.where(n > 0)
    return pd.DataFrame(
        {
            "n_observations": n,
            "mean": (ma + delta * weight).where(n > 0),
            "M2": (a.M2.where(na > 0, 0.0) + b.M2.where(nb > 0, 0.0) + delta**2 * na * weight).where(n > 0, 0.0),
        },
        index=index,
    )


def _extract_moments(sim: Simulation) -> pd.DataFrame | None:
    moments = sim._leg_bid_price_moments
    return None if moments is None else moments.copy()


def _aggregate_moments(summaries: list[GenericSimulationTables]) -> pd.DataFrame | None:
    result = None
    for summary in summaries:
        try:
            # Use the descriptor so lazy PXSIM results are loaded before merging,
            # including custom summary classes that only carry the stored payload.
            frame = SimTabLegBidPriceHistory.leg_bid_price_history._get_raw(summary)
        except MissingDataError:
            continue
        result = _merge_moments(result, frame)
    return result


class _LegBidPriceHistoryItem(SimulationTableItem):
    """Present derived statistics while retaining mergeable moments in storage."""

    @property
    def __doc__(self) -> str | None:
        # A subclass docstring otherwise shadows the base descriptor's API docs.
        return self._doc

    def __get__(
        self, instance: GenericSimulationTables | None, owner: type[GenericSimulationTables] | None = None
    ) -> pd.DataFrame | None | Self:
        if instance is None:
            return self
        moments = super().__get__(instance, owner)
        if moments is None:
            return None
        n = moments.n_observations
        # Build a separate public view: callers cannot accidentally corrupt the
        # raw moments required by worker aggregation or a later archive merge.
        return pd.DataFrame(
            {
                "bid_price_mean": moments["mean"].where(n > 0),
                "bid_price_std_dev": (moments.M2 / (n - 1).where(n > 1)).pow(0.5),
                "n_observations": n,
            },
            index=moments.index,
        )

    def _get_raw(self, instance: GenericSimulationTables) -> pd.DataFrame | None:
        # The base descriptor handles lazy PXSIM loading as well as live results.
        return super().__get__(instance, type(instance))

    def _get_export(self, instance: GenericSimulationTables) -> pd.DataFrame | None:
        return self.__get__(instance, type(instance))


class SimTabLegBidPriceHistory(GenericSimulationTables):
    """Daily bid-price summaries pooled across all retained samples and trials."""

    leg_bid_price_history: pd.DataFrame | None = _LegBidPriceHistoryItem(
        aggregation_func=_aggregate_moments,
        extraction_func=_extract_moments,
        doc="""Daily native bid-price history for every leg, or None when not collected.

        Indexed by ``(leg_id, days_prior)`` with columns ``bid_price_mean``,
        ``bid_price_std_dev`` (pooled sample standard deviation), and
        ``n_observations``. Burn-in samples are excluded. Cells with zero
        observations have undefined mean and standard deviation; cells with one
        observation have undefined standard deviation.

        Collection is enabled by default. Set
        ``simulation_controls.collect_leg_bid_prices=False`` before running to disable it.
        Collection uses core daily-statistics timing and the leg reporting-price
        convention, including sold-out legs. It does not require a database or
        change ``LegBidPriceTracer``. Saved results retain moments so subsequent
        aggregation remains count-weighted and statistically correct.
        """,
    )
