from __future__ import annotations

import io
import pathlib
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
from ortools.linear_solver import pywraplp
from passengersim_core import (
    DynamicProgram,
    SimulationEngine,
)

from passengersim.snapshot.filtering import LegSnapshotFilter

from ._common import RmAction

if TYPE_CHECKING:
    from passengersim.config import Config
    from passengersim.driver import Simulation


class WarnSome:
    def __init__(self, max_times: int = 10, title: str = "") -> None:
        self.n = 0
        self.max_times = max_times
        self.title = title

    def warn(self, message: str, stacklevel: int = 2, **kwargs) -> None:
        if self.n < self.max_times:
            warnings.warn(message, stacklevel=stacklevel + 1, **kwargs)
        self.n += 1

    def __del__(self) -> None:
        if self.title and self.n > self.max_times:
            warnings.warn(
                f"On WarnSome {self.title!r}: {self.n - self.max_times} additional warnings were suppressed",
                stacklevel=2,
            )


class LpDisplacementSolver:
    __slots__ = ("carrier", "solver", "objective", "lp_vars", "constraints", "cabin_code", "warnings", "error_log")

    def __init__(self, sim: SimulationEngine, carrier: str, cabin_code: str, error_log: pathlib.Path | None = None):
        self.carrier = carrier
        self.cabin_code = cabin_code
        self.solver = pywraplp.Solver.CreateSolver("GLOP")
        pywraplp.Solver.SetSolverSpecificParametersAsString(self.solver, "use_dual_simplex:1")
        self.objective = self.solver.Objective()
        self.objective.SetMaximization()
        self.warnings = WarnSome()
        self.error_log = pathlib.Path(error_log) if error_log else None

        # create LP decision variables, which are how many passengers to accept for each path class
        self.lp_vars = {}
        for path in sim.paths.set_filters(carrier=carrier):
            for pc in path.pathclasses:
                if self.cabin_code != "" and pc.cabin_code != self.cabin_code:
                    continue
                name = pc.identifier
                fare = pc.current_tf_adjusted_fare_price(floor=0.0001, nan_to_zero=True)
                # fare = pc.fare.price
                pc_dmd = pc.fcst_mean
                self.lp_vars[name] = x = self.solver.NumVar(0, pc_dmd, name)
                self.objective.SetCoefficient(x, fare)

        # Create capacity constraints
        self.constraints = {}
        for leg in sim.legs.set_filters(carrier=carrier):
            base = leg
            if self.cabin_code != "":
                for cab in leg.cabins:
                    if cab.name == self.cabin_code:
                        base = cab
                        break
            # we set the remaining capacity equal to an epsilon above zero to avoid numerical issues with the solver
            # when legs are sold out, as numerical imprecision can lead the solver to declare the problem infeasible.
            # This should not have a material effect on the results, as the displacement cost should then be very high
            # when the remaining capacity is effectively zero.
            rem_cap = max(self.get_remaining_capacity(base), 0.000001)
            ct = self.solver.Constraint(0, rem_cap, f"Leg-{leg.flt_no}")
            for pc_id in base.pathclass_identifiers:
                ct.SetCoefficient(self.lp_vars[pc_id], 1)
            self.constraints[leg.flt_no] = ct

        # Set a hint that all zeros is feasible
        self.solver.SetHint(self.lp_vars.values(), [0.0] * len(self.lp_vars))

    def update(self, sim: SimulationEngine):
        # update LP decision variables, which are how many passengers to accept for each path class
        # also update LP coefficients, which is the marginal value of taking one more passenger in this pathclass
        for path in sim.paths.set_filters(carrier=self.carrier):
            for pc in path.pathclasses:
                v = self.lp_vars[pc.identifier]
                v.SetUb(pc.fcst_mean)
                self.objective.SetCoefficient(v, pc.current_tf_adjusted_fare_price(floor=0.0001, nan_to_zero=True))

        # Update capacity constraints
        for leg in sim.legs.set_filters(carrier=self.carrier):
            base = leg
            if self.cabin_code != "":
                for cab in leg.cabins:
                    if cab.name == self.cabin_code:
                        base = cab
                        break
            rem_cap = max(self.get_remaining_capacity(base), 0)
            self.constraints[leg.flt_no].SetUb(rem_cap)

    def _write_log(
        self,
        filename: pathlib.Path | io.TextIOBase | None = None,
        with_solution: bool = False,
        with_pickle: bool = False,
    ):
        if isinstance(filename, io.TextIOBase):
            where = filename
        elif isinstance(filename, pathlib.Path | str):
            filename = pathlib.Path(filename)
            filename.parent.mkdir(parents=True, exist_ok=True)
            where = open(filename, "a")
        else:
            return
        try:
            print("## PROBLEM ##", file=where)
            problem_statement = self.solver.ExportModelAsLpFormat(False)
            print(problem_statement.replace("+", "\n    +"), file=where)
            print("\n\n", file=where)
            if with_solution:
                print("## SOLUTION ##", file=where)
                print(f"Objective Value: {self.solver.Objective().Value()}", file=where)
                print("Decision Variables: ", file=where)
                # Loop through all variables registered in the solver
                for k, v in self.lp_vars.items():
                    print(f"  {k}: {v.solution_value()} ({v.lb()} < {v.ub()})", file=where)
                print("Constraint Dual Values: ", file=where)
                for k, v in self.constraints.items():
                    print(f"  {k}: {v.dual_value()}", file=where)
                print("\n\n", file=where)
        finally:
            if isinstance(filename, pathlib.Path):
                where.close()
        if with_pickle and isinstance(filename, pathlib.Path):
            pkl_filename = pathlib.Path(filename).with_suffix(".pkl")
            import dill

            with open(pkl_filename, "wb") as f:
                dill.dump(self.solver, f)

    def get_remaining_capacity(self, base):
        remaining_cap = base.capacity - base.sold
        return remaining_cap

    def solve(self, sim: SimulationEngine, days_prior: int) -> int:
        status = self.solver.Solve()

        if status == pywraplp.Solver.INFEASIBLE:
            self.warnings.warn(
                f"LP Displacement Problem is infeasible: "
                f"carrier={self.carrier}, trial={sim.trial}, sample={sim.sample}, days_prior={days_prior}",
                stacklevel=2,
            )
            if self.error_log:
                self._write_log(
                    self.error_log
                    / f"trial{sim.trial}"
                    / f"sample{sim.sample}"
                    / f"carrier{self.carrier}_cabin{self.cabin_code}_daysprior{days_prior}_infeasible.log"
                )
        elif status == pywraplp.Solver.UNBOUNDED:
            self.warnings.warn(
                f"LP Displacement Problem is unbounded: "
                f"carrier={self.carrier}, trial={sim.trial}, sample={sim.sample}, days_prior={days_prior}",
                stacklevel=2,
            )
            if self.error_log:
                self._write_log(
                    self.error_log
                    / f"trial{sim.trial}"
                    / f"sample{sim.sample}"
                    / f"carrier{self.carrier}_cabin{self.cabin_code}_daysprior{days_prior}_unbounded.log"
                )
        elif status == pywraplp.Solver.ABNORMAL:
            self.warnings.warn(
                f"LP Displacement Problem is abnormal: "
                f"carrier={self.carrier}, trial={sim.trial}, sample={sim.sample}, days_prior={days_prior}",
                stacklevel=2,
            )
            if self.error_log:
                self._write_log(
                    self.error_log
                    / f"trial{sim.trial}"
                    / f"sample{sim.sample}"
                    / f"carrier{self.carrier}_cabin{self.cabin_code}_daysprior{days_prior}_abnormal.log"
                )
        elif status == pywraplp.Solver.MODEL_INVALID:
            self.warnings.warn(
                f"LP Displacement Problem is invalid: "
                f"carrier={self.carrier}, trial={sim.trial}, sample={sim.sample}, days_prior={days_prior}",
                stacklevel=2,
            )
            if self.error_log:
                self._write_log(
                    self.error_log
                    / f"trial{sim.trial}"
                    / f"sample{sim.sample}"
                    / f"carrier{self.carrier}_cabin{self.cabin_code}_daysprior{days_prior}_invalid.log"
                )
        elif status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            # The bid prices used to find displacement adjusted fares are the dual
            # values of the capacity constraints
            num_zero = 0
            for leg in sim.legs.set_filters(carrier=self.carrier):
                # the duals of the constraints are the displacement costs.
                # In theory they should be non-negative by construction, but we
                # enforce that here because the solver can sometimes return tiny
                # negative values due to numerical imprecision.
                leg.displacement = max(self.constraints[leg.flt_no].dual_value(), 0)
                if leg.displacement < 0.01:
                    num_zero += 1
            # print("Num zero displacements = ", num_zero)

        # If the LP is somehow insolvable, we will not change the existing displacement costs.
        # This generally happens closer to departure, probably due to numerical issues with the
        # solver when remaining capacities are tight or sometimes zero.
        return status


class UnbucketedDynamicProgram(RmAction):
    """UDP (Unbucketed Dynamic Programming) is a path-based RM optimization algorithm."""

    requires: set[str] = {"path_forecast"}
    produces: set[str] = {"bid_prices"}
    frequency = "daily_pre_dep"
    snapshot_filter_type = LegSnapshotFilter

    def __init__(
        self,
        *,
        carrier: str = "",
        cfg: Config | None = None,
        store_bid_prices: Literal["daily", "dcp"] = "daily",
        reoptimize: Literal["once", "dcp"] = "dcp",
        arrivals_per_time_slice: float = 0.5,
        min_time_slices_per_dcp: int = 1,
        use_current_bp: bool = False,
        minimum_sample: int = 10,
        normalization_method: int[0, 1, 2, 3, 4] = 0,
        cabins: str | list[str] | None = None,
        capacity_sharing: bool | None = False,
        capacity_sharing_start_dcp_index: int | None = 0,
        capacity_sharing_start_lf: float | None = 0.0,
        snapshot_filters: LegSnapshotFilter | list[LegSnapshotFilter] | None = None,
        error_log: pathlib.Path | None = None,
    ):
        super().__init__(
            carrier=carrier,
            minimum_sample=minimum_sample,
            cfg=cfg,
            snapshot_filters=snapshot_filters,
        )

        self.error_log = error_log
        """
        If problems are encountered, write about them in a log file at this location.

        If not provided, no error log is written.
        """

        self.store_bid_prices: Literal["daily", "dcp"] = store_bid_prices
        """
        The `store_bid_prices` parameter determines how bid prices are used.

        If `store_bid_prices` is set to `daily`, then the bid prices are stored for
        each day prior to departure by dividing each DCP into its component days, and
        then the bid prices are stored for each day, so they can vary for the same level
        of remaining capacity over the days in the DCP.  If set to `dcp`, then the bid
        prices are stored only for each DCP, and are the same for each day within the
        DCP.

        Note that if `store_bid_prices` is set to `daily`, then there must be a call to
        the `udpupdate` RM step in the DAILY section of the RM configuration, which
        updates the bid price algorithm about which day's bid prices to be using in
        offer generation.
        """

        self.reoptimize: Literal["once", "dcp"] = reoptimize
        """
        The `reoptimize` parameter determines how often the UDP is re-run.

        If `reoptimize` is set to `once`, then the UDP is run once for each sample,
        based only the forecast sales by DCP over the entire booking curve.  If set
        to `dcp`, then the UDP is re-run for each DCP, and information on prior sales
        in this sample is used to adjust the displacement costs.
        """

        self.arrivals_per_time_slice: float = arrivals_per_time_slice
        """
        The number of arrivals in each time slice.

        The number of DP simulation time slices in each time period is scaled so the
        total arrivals over all path-classes in one time slice is approximately equal
        to this value. It should not be set to a value greater than 1.0, as this will
        cause the DP to be under-populated.
        """
        if self.arrivals_per_time_slice <= 0.0 or self.arrivals_per_time_slice > 1.0:
            raise ValueError("arrivals_per_time_slice must be in the range (0.0, 1.0]")

        self.min_time_slices_per_dcp: int = min_time_slices_per_dcp
        """
        The minimum number of time slices in each DCP.

        This minimum overrides the `arrivals_per_time_slice` parameter, and ensures
        that each DCP has at least this many time slices.
        """

        self.use_current_bp: bool = use_current_bp
        """
        Experimental.

        Turning this on will assume that bid prices have been set by a prior step
        and just copy them as the displacement values.
        """

        self.normalization_method = normalization_method
        """Normalization is used when the adjusted fares don't sum to the fare value
           0 = No normalization
           1 = Normalize when SUM(adj_fares) > Fare
           2 = Normalize when SUM(adj_fares) < Fare
           3 = Normalize when SUM(adj_fares) != Fare
           4 = Multiply by bid price ratios (based on Juhasz 2023)
           """

        self.cabins = cabins if cabins is not None else []
        """Optional list of cabin codes to optimize.
        If not provided, this step will optimize on the leg as a whole."""

        self.capacity_sharing = capacity_sharing
        """ Capacity sharing flag between cabins.

        When set to True, will use method 3 from Peter Belobaba's presentation.
        Higher cabin(s) will get max of combined cabins or itself alone.
        Lower cabin(s) will get min of combined cabins or itself alone."""

        self.capacity_sharing_start_dcp_index = capacity_sharing_start_dcp_index

        self.capacity_sharing_start_lf = capacity_sharing_start_lf
        """We can optionally turn on capacity sharing when the coach cabin
           reaches a specified load factor.  Based on a suggestion by Darius (PROS)"""

        # init some internal variables
        self._udp_engine = None
        self._lp_engine = {}
        self._max_days_prior = None
        self._dcp_days = None

        dcps = sorted(self.dcps, reverse=True)
        self._max_days_prior = max(dcps)
        self._dcps = dcps
        if self.store_bid_prices == "daily":
            self._dcp_days = -np.diff(dcps, append=0)
            if self._dcp_days.min() <= 0:
                raise ValueError("DCPs must be strictly descending")

    @property
    def udp(self):
        return self._udp

    def run(self, sim: Simulation, days_prior: int):
        if not self.should_run(sim, days_prior):
            return
        if days_prior in self.dcps:
            self._run_dcp(sim, days_prior)
        else:
            self._set_bp_indices(sim, days_prior)

    def _run_dcp(self, sim: Simulation, days_prior: int):
        snapshot_instruction = None
        if sim.snapshot_filters is not None:
            snapshot_filters = sim.snapshot_filters
            for sf in snapshot_filters:
                if sf.type != "udp":
                    continue
                snapshot_instruction = sf.run(sim, carrier=self.carrier)

        # if self._max_days_prior is None:
        #     self.set_dcps(sim.config.dcps)

        # We keep a map of core objects, as the CoreDynamicProgram code caches
        # the data structures it needs for each iteration
        if self._udp_engine is None:
            # TODO: check that we don't need to recreate the engine if the sim changes
            self._udp_engine = DynamicProgram(sim.eng, self.carrier)
            self._udp_engine.initialize()

        dcp_index = self.get_dcp_index(days_prior, allow_between=True)

        if self.reoptimize == "dcp" or (days_prior == self._max_days_prior):
            # Go through the legs and see where we turn on capacity sharing
            for leg in sim.eng.legs.set_filters(carrier=self.carrier):
                if self.capacity_sharing:
                    if self.capacity_sharing_start_lf > 0.01:
                        leg.capacity_sharing = False
                        for cab in leg.cabins:
                            if cab.name == "Y" and cab.sold / cab.capacity >= self.capacity_sharing_start_lf:
                                leg.capacity_sharing = True
                elif self.capacity_sharing_start_dcp_index >= dcp_index:
                    leg.capacity_sharing = True
                else:
                    leg.capacity_sharing = False

            # todo: cache the setup for the LP if needed
            if not self.use_current_bp:
                if len(self.cabins) > 0:
                    for cabin_code in self.cabins:
                        self.solve_carrier_lp_for_leg_fare_displacements(sim.eng, cabin_code, days_prior)
                else:
                    self.solve_carrier_lp_for_leg_fare_displacements(sim.eng, "", days_prior)

            for leg in sim.eng.legs.set_filters(carrier=self.carrier):
                snapshot_instruction = self.apply_snapshot_filters(sim, days_prior, leg)
                if self.capacity_sharing:
                    if self.capacity_sharing_start_lf > 0.01:
                        leg.capacity_sharing = False
                        for cab in leg.cabins:
                            if cab.name == "Y" and cab.sold / cab.capacity >= self.capacity_sharing_start_lf:
                                leg.capacity_sharing = True
                elif self.capacity_sharing_start_dcp_index >= dcp_index:
                    leg.capacity_sharing = True
                else:
                    leg.capacity_sharing = False

                if len(self.cabins) == 0 or self.capacity_sharing:
                    self._run_udp_optimizer(leg, "", snapshot_instruction)
                for cabin_code in self.cabins:
                    self._run_udp_optimizer(leg, cabin_code, snapshot_instruction)

                # leg needs to know current BP index to pick the column when getting bid price
                if self.store_bid_prices == "daily":
                    leg.bp_index = self._max_days_prior - days_prior
                else:
                    leg.bp_index = dcp_index

        else:
            self._set_bp_indices(sim, days_prior)

    def _run_udp_optimizer(self, leg, cabin_code, snapshot_instruction):
        dp = self._udp_engine
        dp.solve_for_leg(
            leg,
            n_dcps=len(self._dcps),
            snapshot_instruction=snapshot_instruction,
            dcp_days=self._dcp_days,
            arrivals_per_time_slice=self.arrivals_per_time_slice,
            min_time_slices_per_dcp=self.min_time_slices_per_dcp,
            normalization_method=self.normalization_method,
            cabin=cabin_code,
        )

    def solve_carrier_lp_for_leg_fare_displacements(self, eng: SimulationEngine, cabin_code="", days_prior: int = -1):
        if cabin_code in self._lp_engine:
            try:
                # it is faster to update the LP than to recreate it
                self._lp_engine[cabin_code].update(eng)
            except KeyError:
                # If we didn't set up the LP correctly with all the pathclasses,
                # then we need to recreate it.
                # TODO: this should not be needed if all pathclasses are initialized correctly
                self._lp_engine[cabin_code] = LpDisplacementSolver(
                    eng, self.carrier, cabin_code, error_log=self.error_log
                )
        else:
            self._lp_engine[cabin_code] = LpDisplacementSolver(eng, self.carrier, cabin_code, error_log=self.error_log)
        self._lp_engine[cabin_code].solve(eng, days_prior)

    def _set_bp_indices(self, sim: Simulation, days_prior: int):
        dcp_index = self.get_dcp_index(days_prior, allow_between=True)
        if self.store_bid_prices == "daily":
            new_bp_index = self._max_days_prior - days_prior
        else:
            new_bp_index = dcp_index
        for leg in sim.eng.legs.set_filters(carrier=self.carrier):
            if self.store_bid_prices == "daily":
                leg.bp_index = new_bp_index
                for cab in leg.cabins:
                    if cab.name in self.cabins:
                        cab.bp_index = new_bp_index


class LegDynamicProgram(UnbucketedDynamicProgram):
    def _run_udp_optimizer(self, leg, cabin_code, snapshot_instruction, starting_dcp_index: int):
        dp = self._udp_engine
        dp.solve_for_leg(
            leg,
            # starting_dcp_index=starting_dcp_index,
            n_dcps=len(self._dcps),
            snapshot_instruction=snapshot_instruction,
            dcp_days=self._dcp_days,
            arrivals_per_time_slice=self.arrivals_per_time_slice,
            min_time_slices_per_dcp=self.min_time_slices_per_dcp,
            normalization_method=self.normalization_method,
            cabin=cabin_code,
            using_buckets=True,
        )

    def _run_dcp(self, sim: Simulation, days_prior: int):
        # We keep a map of core objects, as the CoreDynamicProgram code caches
        # the data structures it needs for each iteration
        if self._udp_engine is None:
            # TODO: check that we don't need to recreate the engine if the sim changes
            self._udp_engine = DynamicProgram(sim.eng, self.carrier)
            self._udp_engine.initialize()

        dcp_index = self.get_dcp_index(days_prior, allow_between=True)

        if self.reoptimize == "dcp" or (days_prior == self._max_days_prior):
            # note: there are no displacement costs when using LegDP.

            starting_dcp_index = self.get_dcp_index(days_prior, allow_between=True)

            for leg in sim.eng.legs.set_filters(carrier=self.carrier):
                snapshot_instruction = self.apply_snapshot_filters(sim, days_prior, leg)

                if len(self.cabins) > 0:
                    for cabin_code in self.cabins:
                        self._run_udp_optimizer(leg, cabin_code, snapshot_instruction, starting_dcp_index)
                else:
                    self._run_udp_optimizer(leg, "", snapshot_instruction, starting_dcp_index)

                # leg needs to know current BP index to pick the column when getting bid price
                if self.store_bid_prices == "daily":
                    leg.bp_index = self._max_days_prior - days_prior
                else:
                    leg.bp_index = dcp_index

        else:
            self._set_bp_indices(sim, days_prior)
