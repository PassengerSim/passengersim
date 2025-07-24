from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import altair as alt

    from passengersim import Simulation, SimulationTables

from collections import defaultdict

import pandas as pd

from .generic import GenericTracer
from .welford import Welford


class LegBidPriceTracer(GenericTracer):
    name: str = "leg_bid_prices"
    bid_prices: dict[int, dict[int, Welford]]

    def __init__(self, leg_ids: list[int], *, priority: int = 1):
        self.leg_ids = leg_ids
        self.priority = priority
        self.reset()

    def reset(self) -> None:
        self.bid_prices = {leg_id: defaultdict(Welford) for leg_id in self.leg_ids}

    def fresh(self) -> LegBidPriceTracer:
        return LegBidPriceTracer(self.leg_ids, priority=self.priority)

    def attach(self, sim: Simulation) -> None:
        sim.daily_callback(self.fresh())

    def __call__(self, sim: Simulation, days_prior: int):
        if sim.sim.sample < sim.sim.burn_samples:
            return
        for leg_id in self.leg_ids:
            self.bid_prices[leg_id][days_prior].update(sim.legs[leg_id].report_bid_price)

    def finalize(self) -> pd.DataFrame:
        m = pd.DataFrame(
            {
                leg_id: {days_prior: welford.mean for days_prior, welford in welfords.items()}
                for leg_id, welfords in self.bid_prices.items()
            }
        ).rename_axis(columns="leg_id", index="days_prior")
        s = pd.DataFrame(
            {
                leg_id: {days_prior: welford.sample_std_dev for days_prior, welford in welfords.items()}
                for leg_id, welfords in self.bid_prices.items()
            }
        ).rename_axis(columns="leg_id", index="days_prior")
        return pd.concat([m, s], keys=["mean", "std_dev"], names=["statistic"])


def _fig_traced_bid_prices(
    summary: SimulationTables,
    *,
    std_dev: bool = True,
    include: tuple[str] | None = None,
    raw_df: bool = False,
    kind: Literal["leg", "path"] = "leg",
    ids: tuple[int] | None = None,
) -> alt.Chart:
    from passengersim.contrast import Contrast  # prevent circular import

    if isinstance(summary, Contrast):
        if include is not None:
            summary = {k: v for k, v in summary.items() if k in include}
        dfs = {k: _fig_traced_bid_prices(v, raw_df=True, kind=kind) for k, v in summary.items()}
        df = pd.concat(dfs, names=["source"]).reset_index()
    else:
        df = getattr(summary.callback_data, f"{kind}_bid_prices")
        df = df.stack().unstack("statistic").reset_index()

    if ids is not None:
        df = df.query(f"{kind}_id in @ids")

    if raw_df:
        return df

    import altair as alt

    chart = alt.Chart(df)
    fig = chart.mark_line().encode(
        x=alt.X("days_prior:Q").scale(reverse=True),
        y=alt.Y("mean:Q", title="Mean Bid Price"),
        color=f"{kind}_id:N",
    )
    if "source" in df.columns:
        fig = fig.encode(strokeDash="source:N", strokeWidth="source:N")
    if std_dev:
        fig1 = chart.mark_line().encode(
            x=alt.X("days_prior:Q").scale(reverse=True),
            y=alt.Y("std_dev:Q", title="Std Dev Bid Price"),
            color=f"{kind}_id:N",
        )
        if "source" in df.columns:
            fig1 = fig1.encode(strokeDash="source:N", strokeWidth="source:N")
        fig = fig | fig1

    return fig


def fig_leg_bid_prices(
    summary: SimulationTables,
    *,
    std_dev: bool = True,
    include: tuple[str] | None = None,
    raw_df: bool = False,
    ids: tuple[int] | None = None,
) -> alt.Chart:
    return _fig_traced_bid_prices(summary, std_dev=std_dev, include=include, raw_df=raw_df, kind="leg", ids=ids)


class PathBidPriceTracer(GenericTracer):
    name: str = "path_bid_prices"
    bid_prices: dict[int, dict[int, Welford]]

    def __init__(self, path_ids: list[int], *, priority=1):
        self.path_ids = path_ids
        self.priority = priority
        self.reset()

    def reset(self) -> None:
        self.bid_prices = {path_id: defaultdict(Welford) for path_id in self.path_ids}

    def fresh(self) -> PathBidPriceTracer:
        return PathBidPriceTracer(self.path_ids, priority=self.priority)

    def attach(self, sim: Simulation) -> None:
        sim.daily_callback(self.fresh())

    def __call__(self, sim: Simulation, days_prior: int):
        if sim.sim.sample < sim.sim.burn_samples:
            return
        for path_id in self.path_ids:
            self.bid_prices[path_id][days_prior].update(sim.paths.select(path_id=path_id).report_total_bid_price())

    def finalize(self) -> pd.DataFrame:
        m = pd.DataFrame(
            {
                path_id: {days_prior: welford.mean for days_prior, welford in welfords.items()}
                for path_id, welfords in self.bid_prices.items()
            }
        ).rename_axis(columns="path_id", index="days_prior")
        s = pd.DataFrame(
            {
                path_id: {days_prior: welford.sample_std_dev for days_prior, welford in welfords.items()}
                for path_id, welfords in self.bid_prices.items()
            }
        ).rename_axis(columns="path_id", index="days_prior")
        return pd.concat([m, s], keys=["mean", "std_dev"], names=["statistic"])


def fig_path_bid_prices(
    summary: SimulationTables,
    *,
    std_dev: bool = True,
    include: tuple[str] | None = None,
    raw_df: bool = False,
    ids: tuple[int] | None = None,
) -> alt.Chart:
    return _fig_traced_bid_prices(summary, std_dev=std_dev, include=include, raw_df=raw_df, kind="path", ids=ids)
