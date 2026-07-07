from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from passengersim_core import EMSR

from passengersim.snapshot.filtering import LegSnapshotFilter

from ._common import RmAction

if TYPE_CHECKING:
    from passengersim.config import Config
    from passengersim.driver import Simulation


class ExpectedMarginalSeatRevenue(RmAction):
    """
    EMSR (Expected Marginal Seat Revenue) is a leg-based optimization algorithm.

    The EMSR (Expected Marginal Seat Revenue) algorithm is a widely adopted
    heuristic for capacity allocation in revenue management, primarily used to
    determine how many seats to protect for higher-fare classes in a restricted
    product marketplace. It works by calculating a booking limit for each fare
    class, starting with the lowest, to maximize expected revenue.

    The "A" algorithm compares each class individually, and is not generally used
    in practice. The "B" algorithm, often called EMSR-B, aggregates displaced
    low-fare passengers. This method uses Littlewood's rule with the aggregate
    demand and average fare to statistically determine optimal protection levels
    for the remaining higher-priced inventory, effectively balancing the risk of
    selling a seat cheaply now versus the potential of selling it at a higher
    price later.

    Applying the EMSR algorithm requires a forecast of leg demand by fare class.
    Protection levels are updated at each decision control point (DCP) based on the
    latest sales and demand forecasts, allowing for dynamic adjustment to changing
    booking patterns.
    """

    requires: set[str] = {"leg_forecast"}
    produces: set[str] = {"bucket_allocations"}
    frequency = "dcp"
    snapshot_filter_type = LegSnapshotFilter

    OPT = EMSR()  # singleton EMSR optimizer

    def __init__(
        self,
        *,
        variant: Literal["a", "b", "c"] = "b",
        carrier: str = "",
        cabins: str | list[str] | None = None,
        minimum_sample: int = 10,
        snapshot_filters: LegSnapshotFilter | list[LegSnapshotFilter] | None = None,
        cfg: Config | None = None,
    ):
        super().__init__(
            carrier=carrier,
            minimum_sample=minimum_sample,
            cfg=cfg,
            snapshot_filters=snapshot_filters,
        )

        self.variant = variant
        """EMSR variant to use: "a", "b", or "c"."""

        if self.variant not in ["a", "b", "c"]:
            raise ValueError(f"Unknown EMSR variant {self.variant!r}")

        self.cabins = cabins
        """Optional list of cabin codes to optimize.

        If not provided, this tool will optimize on the leg as a whole."""

    def run(self, sim: Simulation, days_prior: int):
        if not self.should_run(sim, days_prior):
            return

        for leg in sim.eng.legs.set_filters(carrier=self.carrier):
            snapshot_instruction = self.apply_snapshot_filters(sim, days_prior, leg)
            if self.variant == "a":
                raise NotImplementedError("EMSR-A variant is not implemented.")
            elif self.variant == "b":
                if self.cabins is None:
                    leg.emsrb(debug=snapshot_instruction)
                else:
                    for cabin in leg.cabins:
                        if cabin.name in self.cabins:
                            self.OPT.emsrb(cabin)
            elif self.variant == "c":
                raise NotImplementedError("EMSR-C variant is not implemented.")
                # leg.emsrc(False)
            else:
                raise ValueError(f"Unknown EMSR variant {self.variant!r}")
