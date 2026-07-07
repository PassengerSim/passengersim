from __future__ import annotations

from typing import TYPE_CHECKING

from passengersim_core import ProBP

from passengersim.snapshot.filtering import NetworkSnapshotFilter

from ._common import RmAction

if TYPE_CHECKING:
    from passengersim.config import Config
    from passengersim.driver import Simulation


class ProbabilisticBidPrice(RmAction):
    """
    ProBP (ProbabilisticBidPrice) is a path-based optimization algorithm.
    """

    requires: set[str] = {"path_forecast"}
    produces: set[str] = {"bid_prices"}
    frequency = "daily_pre_dep"
    snapshot_filter_type = NetworkSnapshotFilter

    def __init__(
        self,
        *,
        carrier: str = "",
        cabins: str | list[str] | None = None,
        minimum_sample: int = 10,
        cfg: Config | None = None,
        capacity_sharing: bool | None = False,
        capacity_sharing_start_dcp_index: int | None = 0,
        capacity_sharing_start_lf: float | None = 0.0,
        use_adjusted_fares: bool = False,
        bid_price_vector: bool | None = False,
        maxiter: int = 10,
        use_sub_bp: bool = False,
        snapshot_filters: NetworkSnapshotFilter | list[NetworkSnapshotFilter] | None = None,
    ):
        super().__init__(
            carrier=carrier,
            minimum_sample=minimum_sample,
            cfg=cfg,
            snapshot_filters=snapshot_filters,
        )

        self._pro_bp_engine = None

        self.cabins = cabins
        """Optional list of cabin codes to optimize.

        If not provided, this tool will optimize on the leg as a whole."""

        self.capacity_sharing = capacity_sharing
        """ Capacity sharing flag between cabins.

        When set to True, will use method 3 from Peter Belobaba's presentation.
        Higher cabin(s) will get max of combined cabins or itself alone.
        Lower cabin(s) will get min of combined cabins or itself alone."""

        self.capacity_sharing_start_dcp_index = capacity_sharing_start_dcp_index

        self.capacity_sharing_start_lf = capacity_sharing_start_lf
        """We can optionally turn on capacity sharing when the coach cabin
           reaches a specified load factor.  Based on a suggestion by Darius (PROS)"""

        self.use_adjusted_fares = use_adjusted_fares
        """
        If True, ProBP will use the adjusted fares for the optimization.

        The default is False, which means that ProBP will use the original fares.  This
        should be set to True if fare adjustment is being used for this carrier.
        """

        self.bid_price_vector = bid_price_vector
        """
        If True, we create a bid price vector in ProBP,
        rather than just keep a constant bid-price untiol daily re-optimization
        """

        self.maxiter = maxiter
        """
        The maximum number of iterations to run ProBP.

        If the algorithm has not converged by the time this number of iterations has
        been reached, it will stop and return the current results.
        """

        self.use_sub_bp: bool = use_sub_bp
        """Whether to use SubBP (True) or ProBP (False)."""

    def rm_engine(self, sim: Simulation) -> ProBP:
        # We a reference to a ProBP object, as the CoreProBP code caches
        # the data structures it needs for each iteration
        if self._pro_bp_engine is None:
            engine = ProBP(sim.eng, self.carrier)
            if self.use_sub_bp:
                engine.use_sub_bp = True
            self._pro_bp_engine = engine
        return self._pro_bp_engine

    def run(self, sim: Simulation, days_prior: int):
        if not self.should_run(sim, days_prior):
            return

        # Make sure each sample is initialized
        if sim.eng.last_dcp_index <= 1:
            for leg in sim.eng.legs.set_filters(carrier=self.carrier):
                leg.capacity_sharing = False

        if self.capacity_sharing:
            if self.capacity_sharing_start_dcp_index >= sim.eng.last_dcp_index:
                for leg in sim.eng.legs.set_filters(carrier=self.carrier):
                    leg.capacity_sharing = True
            elif self.capacity_sharing_start_lf > 0.01:
                for leg in sim.eng.legs.set_filters(carrier=self.carrier):
                    for cab in leg.cabins:
                        if cab.name == "Y" and cab.sold / cab.capacity >= self.capacity_sharing_start_lf:
                            leg.capacity_sharing = True

        z = self.rm_engine(sim)

        # Update the decision fares if needed
        dcp_index = self.get_dcp_index(days_prior, allow_between=True)
        if self.use_adjusted_fares:
            z.update_decision_fares(dcp_index)

        snapshot_instruction = self.apply_snapshot_filters(sim, days_prior)

        # num_cabins = len(self.cabins) if self.cabins is not None else 0
        debug_output = z.run(
            "", maxiter=self.maxiter, bid_price_vector=self.bid_price_vector, snapshot_instruction=snapshot_instruction
        )
        if self.cabins:
            for c in self.cabins:
                debug_output = z.run(
                    c,
                    maxiter=self.maxiter,
                    bid_price_vector=self.bid_price_vector,
                    snapshot_instruction=snapshot_instruction,
                )

        if debug_output:
            print(debug_output)
