from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from passengersim import Simulation

from collections import defaultdict

import pandas as pd

from .generic import GenericTracer
from .welford import Welford


class LegBidPriceTracer(GenericTracer):
    bid_prices: dict[int, dict[int, Welford]]

    def __init__(self, leg_ids: list[int]):
        self.leg_ids = leg_ids
        self.reset()

    def reset(self) -> None:
        self.bid_prices = {leg_id: defaultdict(Welford) for leg_id in self.leg_ids}

    def attach(self, sim: Simulation) -> None:
        cls = type(self)
        tracer = cls(self.leg_ids)
        sim.daily_callback(tracer)

    def __call__(self, sim: Simulation, days_prior: int):
        for leg_id in self.leg_ids:
            self.bid_prices[leg_id][days_prior].update(sim.legs[leg_id].bid_price)

    def finalize(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                leg_id: {
                    days_prior: welford.mean for days_prior, welford in welfords.items()
                }
                for leg_id, welfords in self.bid_prices.items()
            }
        ).rename_axis(columns="leg_id", index="days_prior")
