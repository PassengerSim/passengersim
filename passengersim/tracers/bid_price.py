from __future__ import annotations

from typing import TYPE_CHECKING

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

    def __init__(self, leg_ids: list[int]):
        self.leg_ids = leg_ids
        self.reset()

    def reset(self) -> None:
        self.bid_prices = {leg_id: defaultdict(Welford) for leg_id in self.leg_ids}

    def fresh(self) -> LegBidPriceTracer:
        return LegBidPriceTracer(self.leg_ids)

    def attach(self, sim: Simulation) -> None:
        sim.daily_callback(self.fresh())

    def __call__(self, sim: Simulation, days_prior: int):
        if sim.sim.sample < sim.sim.burn_samples:
            return
        for leg_id in self.leg_ids:
            self.bid_prices[leg_id][days_prior].update(
                sim.legs[leg_id].report_bid_price
            )

    def finalize(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                leg_id: {
                    days_prior: welford.mean for days_prior, welford in welfords.items()
                }
                for leg_id, welfords in self.bid_prices.items()
            }
        ).rename_axis(columns="leg_id", index="days_prior")


def fig_leg_bid_prices(summary: SimulationTables, *, raw_df: bool = False) -> alt.Chart:
    from passengersim.contrast import Contrast  # prevent circular import

    if isinstance(summary, Contrast):
        dfs = {k: fig_leg_bid_prices(v, raw_df=True) for k, v in summary.items()}
        df = pd.concat(dfs, names=["source"]).reset_index()
    else:
        df = summary.callback_data.leg_bid_prices
        df = df.stack().rename("bid_price").reset_index()

    if raw_df:
        return df

    import altair as alt

    fig = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("days_prior:Q").scale(reverse=True),
            y=alt.Y("bid_price:Q"),
            color="leg_id:N",
        )
    )
    if "source" in df.columns:
        fig = fig.encode(strokeDash="source:N", strokeWidth="source:N")

    return fig
