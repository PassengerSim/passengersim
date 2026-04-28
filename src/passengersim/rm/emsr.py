from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Literal

from passengersim_core import EMSR

from passengersim.config.snapshot_filter import SnapshotFilter
from passengersim.snapshot import get_snapshot_instruction

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
    frequency = "dcp"

    OPT = EMSR()  # singleton EMSR optimizer

    def __init__(
        self,
        *,
        variant: Literal["a", "b", "c"] = "b",
        carrier: str = "",
        cabins: str | list[str] | None = None,
        minimum_sample: int = 10,
        snapshots: Collection[SnapshotFilter | dict] = (),
        cfg: Config | None = None,
    ):
        super().__init__(
            carrier=carrier,
            minimum_sample=minimum_sample,
            cfg=cfg,
        )

        self.variant = variant
        """EMSR variant to use: "a", "b", or "c"."""

        if self.variant not in ["a", "b", "c"]:
            raise ValueError(f"Unknown EMSR variant {self.variant!r}")

        self.cabins = cabins
        """Optional list of cabin codes to optimize.

        If not provided, this tool will optimize on the leg as a whole."""

        self.snapshots = []
        """Optional list of snapshots to grap when running this action."""

        # populate snapshots
        for i in snapshots:
            if not isinstance(i, SnapshotFilter):
                i = SnapshotFilter(**i)
            self.snapshots.append(i)

    def run(self, sim: Simulation, days_prior: int):
        if not self.should_run(sim, days_prior):
            return

        for leg in sim.eng.legs.set_filters(carrier=self.carrier):
            snap_instruct = False
            if self.snapshots:
                snap_instruct = get_snapshot_instruction(
                    sim.eng,
                    filters=self.snapshots,
                )
            if self.variant == "a":
                raise NotImplementedError("EMSR-A variant is not implemented.")
            elif self.variant == "b":
                if self.cabins is None:
                    leg.emsrb(debug=snap_instruct)
                else:
                    for cabin in leg.cabins:
                        if cabin.name in self.cabins:
                            self.OPT.emsrb(cabin)
            elif self.variant == "c":
                raise NotImplementedError("EMSR-C variant is not implemented.")
                # leg.emsrc(False)
            else:
                raise ValueError(f"Unknown EMSR variant {self.variant!r}")
