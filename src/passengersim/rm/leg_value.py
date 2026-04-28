from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from ._common import RmAction

if TYPE_CHECKING:
    from passengersim.config import Config
    from passengersim.driver import Simulation


class LegValue(RmAction):
    produces: set[str] = {"leg_value"}
    frequency = "begin_sample"

    def __init__(
        self,
        *,
        carrier: str = "",
        minimum_sample: int = 3,
        cfg: Config | None = None,
        algorithm: Literal["bottom_up", "top_down"] = "bottom_up",
        minimum_pct_separation: float = 0.05,
        frequency: Literal["dcp", "daily", "begin_sample"] = "dcp",
    ):
        super().__init__(
            carrier=carrier,
            minimum_sample=minimum_sample,
            cfg=cfg,
        )

        self.algorithm: Literal["bottom_up", "top_down"] = algorithm
        """Algorithm to use for leg value correction.

        Corrections may be needed to address fare inversions. The `bottom_up`
        algorithm will start with the lowest bucket and work up, pushing fares
        upwards if needed to maintain ordering and ensure the indicated minimum
        percentage separation. The `top_down` algorithm will start with the highest
        bucket and work down, pushing fares downwards if needed.
        """

        self.minimum_pct_separation: float = minimum_pct_separation
        """Minimum percentage separation between bucket values.

        If the computed values are inverted or too close to each other, many
        optimization algorithms will fail.  This parameter ensures that the
        values are separated by at least this percentage at each step.
        """

        self.frequency: Literal["dcp", "daily", "begin_sample"] = frequency
        """How often to run this step."""

    def run(self, sim: Simulation, days_prior: int):
        if not self.should_run(sim, days_prior):
            return
        if self.algorithm == "bottom_up":
            for leg in sim.eng.legs.set_filters(carrier=self.carrier):
                previous_value = 0.0
                # assume buckets are sorted by naive value from high to low
                for bkt in reversed(leg.buckets):
                    bkt.fcst_revenue = max(bkt.prorated_value, previous_value * (1 + self.minimum_pct_separation))
                    previous_value = bkt.fcst_revenue
        elif self.algorithm == "top_down":
            for leg in sim.eng.legs.set_filters(carrier=self.carrier):
                previous_value = np.inf
                for bkt in leg.buckets:
                    bkt.fcst_revenue = min(bkt.prorated_value, previous_value * (1 - self.minimum_pct_separation))
                    previous_value = bkt.fcst_revenue
