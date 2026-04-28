#
# Several LP approaches to RM optimization
#
# Alan W
# (c) PassengerSim LLC, January 2026
#

from __future__ import annotations

import random
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from statistics import NormalDist
from typing import TYPE_CHECKING

from ortools.linear_solver import pywraplp
from passengersim_core import DynamicProgram

from ._common import RmAction

if TYPE_CHECKING:
    from passengersim.config import Config
    from passengersim.core import SimulationEngine
    from passengersim.driver import Simulation


class LinearProgramming(RmAction):
    """
    Uses various LP implementations as an RM step, either for displacement computations
    or for bid price.
    """

    requires: set[str] = {"path_forecast"}
    frequency = "dcp"
    lp = {}

    def __init__(
        self,
        *,
        carrier: str = "",
        cfg: Config | None = None,
        algorithm: str = "piecewise",
        epsilon: float = 0.01,
        num_pieces: float = 25,
        max_std_dev: float = 2.5,
        debug: bool = False,
    ):
        super().__init__(
            carrier=carrier,
            cfg=cfg,
        )

        self.algorithm = algorithm
        self.epsilon = epsilon
        self.carrier = carrier
        self.epsilon = epsilon
        self.num_pieces = num_pieces
        self.max_std_dev = max_std_dev
        self.debug = debug

    def run(self, sim: Simulation, days_prior: int):
        if not self.should_run(sim, days_prior):
            return

        if self.carrier not in self.lp:
            if self.algorithm.lower() in ["deterministic", "dlp"]:
                s = DLP(self.carrier, self.debug)
            elif self.algorithm.lower() in ["piecewise", "pnlp"]:
                self.frequency = "daily_pre_dep"
                s = LpPiecewiseSolver(self.carrier, self.debug)
            elif self.algorithm.lower() in ["piecewise2", "rithvik"]:
                self.frequency = "daily_pre_dep"
                s = LpPiecewise2(self.carrier, 0.01, self.debug)
            elif self.algorithm.lower() == "ssa":
                s = StochasticSampleSolver(self.carrier, self.debug)
            elif self.algorithm.lower() == "ssblp":
                raise ValueError("Not implemented yet")
            else:
                raise ValueError("Unknown algorithm")
            s.initialize(sim.eng)
            self.lp[self.carrier] = s
        else:
            self.lp[self.carrier].update(sim.eng)

        self.lp[self.carrier].solve(sim.eng)


# ##########################################################################################
class LpBase(ABC):
    """Some common code, used by each implementation.
    Also forces each implementation to have a standard interface"""

    @abstractmethod
    def initialize(self, eng: SimulationEngine):
        pass

    @abstractmethod
    def update(self, eng: SimulationEngine):
        pass

    @abstractmethod
    def solve(self, sim: SimulationEngine, debug=False):
        pass

    def test_status(self, status):
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return True
        elif status == pywraplp.Solver.INFEASIBLE:
            msg = "LP Problem is infeasible"
        elif status == pywraplp.Solver.UNBOUNDED:
            msg = "LP Problem is unbounded"
        elif status == pywraplp.Solver.ABNORMAL:
            msg = "LP Problem is abnormal"
        elif status == pywraplp.Solver.MODEL_INVALID:
            msg = "LP Displacement Problem is invalid"
        else:
            msg = "LP Unknown status - this should never happen !!!"
        warnings.warn(f"{msg}: carrier={self.carrier}, sample={self.sim.sample}", stacklevel=2)
        return False


# ##########################################################################################


class DLP(LpBase):
    """Deterministic LP, can be used to get displacement for UDP
    Can also set bid-prices as a demo of why deterministic LP isn't used
    for this purpose in any real RM system :-)"""

    __slots__ = ("carrier", "solver", "objective", "lp_vars", "constraints", "cabin_code")

    def __init__(self, sim: SimulationEngine, carrier: str, cabin_code: str):
        self.carrier = carrier
        self.cabin_code = cabin_code
        self.solver = pywraplp.Solver.CreateSolver("GLOP")
        pywraplp.Solver.SetSolverSpecificParametersAsString(self.solver, "use_dual_simplex:1")
        self.objective = self.solver.Objective()
        self.objective.SetMaximization()

        # create LP decision variables, which are how many passengers to accept for each path class
        self.lp_vars = {}
        for path in sim.paths.set_filters(carrier=carrier):
            for pc in path.pathclasses:
                if self.cabin_code != "" and pc.cabin_code != self.cabin_code:
                    continue
                name = pc.identifier
                fare = pc.fare_price
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
            rem_cap = max(self.get_remaining_capacity(base), 0)
            ct = self.solver.Constraint(0, rem_cap, f"Leg-{leg.flt_no}")
            for pc_id in base.pathclass_identifiers:
                ct.SetCoefficient(self.lp_vars[pc_id], 1)
            self.constraints[leg.flt_no] = ct

    def update(self, eng: SimulationEngine):
        # update LP decision variables, which are how many passengers to accept for each path class
        for path in eng.paths.set_filters(carrier=self.carrier):
            for pc in path.pathclasses:
                self.lp_vars[pc.identifier].SetUb(pc.fcst_mean)

        # Create capacity constraints
        for leg in eng.legs.set_filters(carrier=self.carrier):
            base = leg
            if self.cabin_code != "":
                for cab in leg.cabins:
                    if cab.name == self.cabin_code:
                        base = cab
                        break
            rem_cap = max(self.get_remaining_capacity(base), 0)
            self.constraints[leg.flt_no].SetUb(rem_cap)

    def get_remaining_capacity(self, base):
        remaining_cap = base.capacity - base.sold
        return remaining_cap

    def solve(self, sim: SimulationEngine):
        status = self.solver.Solve()
        if self.test_status(status):
            # The bid prices used to find displacement adjusted fares are the dual
            # values of the capacity constraints
            num_zero = 0
            for leg in sim.legs.set_filters(carrier=self.carrier):
                # the duals of the contraints are the displacement costs.
                # In theory they should be non-negative by construction, but we
                # enforce that here because the solver can sometimes return tiny
                # negative values due to numerical imprecision.
                leg.displacement = max(self.constraints[leg.flt_no].dual_value(), 0)
                if leg.displacement < 0.01:
                    num_zero += 1
            # print("Num zero displacements = ", num_zero)
            return

        # If the LP is infeasible, we need to set the displacement costs to zero
        for leg in sim.legs.set_filters(carrier=self.carrier):
            leg.displacement = 0


# ##########################################################################################
class LpPiecewiseSolver(LpBase):
    """Just fiddling with an approach to approximate the non-linear stochastic problem.
    Break each demand into pieces, and price them at the expected marginal revenue
    NOTES:
        - We don't need an "allocation <= demand" constraint, as demand is effectively unbounded
          but the marginal revenue is monotonic descending, asymptotic to zero and so the
          pieces will get chosen in order until leg capacity is filled.
    """

    __slots__ = (
        "carrier",
        "solver",
        "objective",
        "lp_vars",
        "constraints",
        "epsilon",
        "num_pieces",
        "max_std_dev",
        "debug",
    )

    def __init__(self, carrier: str, _debug=False):
        self.epsilon = 0.01
        self.carrier = carrier
        self.num_pieces = 25
        self.max_std_dev = 2.5
        self.debug = _debug

    def initialize(self, eng: SimulationEngine):
        self.solver = pywraplp.Solver.CreateSolver("GLOP")
        self.objective = self.solver.Objective()
        self.objective.SetMaximization()

        tmp = DynamicProgram(eng, self.carrier)
        tmp.initialize()  # This step sets up the PcPtr structures, will refactor later to make this a separate call

        # create LP decision variables, which are how many passengers to accept for each path class
        self.lp_vars = {}
        if self.debug:
            print("********** Setup decision variables **********")

        for path in eng.paths:  # .set_filters(carrier=self.carrier):
            if path.carrier_name != self.carrier:
                continue
            if self.debug:
                print("Path:", path)

            for pc in path.pathclasses:
                fare = pc.fare_price
                pc_dmd = pc.fcst_mean
                pc_std_dev = pc.fcst_std_dev
                var_max = pc_dmd + self.max_std_dev * pc_std_dev
                piece_size = var_max / self.num_pieces
                lh_rev = fare
                for piece in range(self.num_pieces):
                    # What the probability that demand is at least (piece * piece_size)?
                    if pc_dmd < self.epsilon or pc_std_dev < self.epsilon:
                        prob = 0.0
                    else:
                        prob = 1.0 - NormalDist(mu=pc_dmd, sigma=pc_std_dev).cdf(piece * piece_size)
                    # As we step along the EMSR curve, compute the LH and RH of the piece, then take the average
                    rh_rev = fare * prob
                    avg_rev = (lh_rev + rh_rev) / 2.0
                    lh_rev = rh_rev
                    name = f"{pc.identifier}-{piece}"
                    self.lp_vars[name] = x = self.solver.NumVar(0, piece_size, name)
                    if self.debug:
                        print(f"    Var: {name}, size={piece_size}, rev={avg_rev}")
                    self.objective.SetCoefficient(x, avg_rev)

        # Create capacity constraints
        if self.debug:
            print("CAPACITY CONSTRAINTS")
        self.constraints = {}

        for leg in eng.legs:
            if leg.carrier_name != self.carrier:
                continue
            ct = self.solver.Constraint(0, leg.capacity - leg.sold, f"Leg-{leg.flt_no}")
            for pc_id in leg.pathclass_identifiers:
                # Add up all the pieces on the leg
                for piece in range(self.num_pieces):
                    name = f"{pc_id}-{piece}"
                    ct.SetCoefficient(self.lp_vars[name], 1)
                    if self.debug:
                        print(f"    Leg={leg}, name={name}")
            self.constraints[leg.flt_no] = ct

    def update(self, eng: SimulationEngine):
        self.initialize(eng)

    def solve(self, sim: SimulationEngine, debug=False):
        _result_status = self.solver.Solve()
        if self.test_status(_result_status):
            # The bid prices are the dual values of the capacity constraints
            for leg in sim.legs:
                if leg.carrier_name != self.carrier:
                    continue
                # The duals of the constraints are the displacement costs.
                # In theory, they should be non-negative by construction, but we
                # enforce that here because the solver can sometimes return tiny
                # negative values due to numerical imprecision.
                leg.displacement = max(self.constraints[leg.flt_no].dual_value(), 0)
                leg.bid_price = leg.displacement
                if debug:
                    print(f"Leg: {leg}, bp={leg.displacement}")


# ##########################################################################################
class StochasticSampleSolver(LpBase):
    """Testing a sample-based LP solver.
    The x variables correspond to each sample
    The y variables are the actual (nested) controls
    We have constraints so that the x variables nest in the y variables
    Based on [Kleywegt 2002]
    """

    __slots__ = (
        "carrier",
        "solver",
        "objective",
        "lp_vars",
        "constraints",
        "num_samples",
        "debug",
    )

    def __init__(self, carrier: str, num_samples=5, debug=False):
        self.carrier = carrier
        self.num_samples = num_samples
        self.debug = debug

    def initialize(self, sim: SimulationEngine):
        self.solver = pywraplp.Solver.CreateSolver("GLOP")
        self.objective = self.solver.Objective()
        self.objective.SetMaximization()

        tmp = DynamicProgram(sim, self.carrier)
        tmp.initialize(False)  # Sets up the PcPtr structures, will refactor later to make this a separate call

        # create LP decision variables, which are how many passengers to accept for each path class
        self.lp_vars = {}
        if self.debug:
            print(f"********** Sample = {sim.sample}, setup decision variables **********")
        for path in sim.paths:  # .set_filters(carrier=self.carrier):
            if path.carrier_name != self.carrier:
                continue
            if self.debug:
                print("Path:", path)

            p_id = f"{path.carrier_name}_{path.orig}_{path.dest}_{path.path_id}"
            c_name = f"PathRev-{p_id}"
            rev_c = self.solver.Constraint(0, self.solver.infinity(), c_name)

            pc_vars = []
            sample_vars = defaultdict(list)
            for pc in path.pathclasses:
                y_name = "y_" + pc.identifier
                fare = pc.fare_price
                pc_dmd = pc.fcst_mean
                self.lp_vars[y_name] = y = self.solver.NumVar(0, pc_dmd, y_name)
                if self.debug:
                    print("  Set Obj:", y, fare)
                self.objective.SetCoefficient(y, 0.0)
                rev_c.SetCoefficient(y, fare)
                pc_vars.append(y)

                # Sample variables
                rev = fare / self.num_samples
                for s in range(self.num_samples):
                    sample_demand = max(random.normalvariate(mu=pc_dmd, sigma=pc.fcst_std_dev), 0.0)
                    x_name = f"x_{s}_{pc.identifier}"
                    self.lp_vars[x_name] = x = self.solver.NumVar(0, sample_demand, x_name)
                    sample_vars[s].append(x)
                    if self.debug:
                        print("  Set Obj", x, rev, pc_dmd, round(sample_demand, 2))
                    self.objective.SetCoefficient(x, rev)
                    if self.debug:
                        print("  Set Constraint", c_name, x, -1.0 * rev)
                    rev_c.SetCoefficient(x, -1.0 * rev)

            # Nesting constraints
            # Suppose we have Y0, Y1 & Y2, we do this for each sample:
            #   x_Y2 <= y_Y2
            #   x_Y2 + x_Y1 <= y_Y2 + y_Y1
            #   x_Y2 + x_Y1 + x_Y0 <= y_Y2 + y_Y1 + y_Y0
            for s in range(self.num_samples):
                for j in range(len(pc_vars)):
                    nest_name = f"SampleNest-{p_id}-{s}-{j}"
                    if self.debug:
                        print("  Create sample nest constraint:", nest_name)
                    nest_c = self.solver.Constraint(0, self.solver.infinity(), nest_name)
                    for i in range(j, len(pc_vars)):
                        x = sample_vars[s][i]
                        y = pc_vars[i]
                        if self.debug:
                            print("    Coeffs:", x, y)
                        nest_c.SetCoefficient(x, -1.0)
                        nest_c.SetCoefficient(y, 1.0)

        # Create capacity constraints
        self.constraints = {}
        for leg in sim.legs.set_filters(carrier=self.carrier):
            rem_cap = leg.capacity - leg.sold
            ct_name = f"Leg-{leg.flt_no}"
            if self.debug:
                print("Capacity constraint", ct_name, rem_cap)
            ct = self.solver.Constraint(0, rem_cap, ct_name)
            for pc_id in leg.pathclass_identifiers:
                name = "y_" + pc_id
                if self.debug:
                    print("  Adding variable", name)
                ct.SetCoefficient(self.lp_vars[name], 1)
            self.constraints[leg.flt_no] = ct

    def update(self, eng: SimulationEngine):
        self.initialize(eng)

    def solve(self, sim: SimulationEngine):
        status = self.solver.Solve()
        if self.test_status(status):
            for leg in sim.legs.set_filters(carrier=self.carrier):
                dual = self.constraints[leg.flt_no].dual_value()
                leg.displacement = max(dual, 0)
                leg.bid_price = leg.displacement
                print(f"Dual for {leg} = {round(dual, 2)}")
            for k, var in self.lp_vars.items():
                if k[0] != "x":
                    print(k, round(var.solution_value(), 2))


# ##########################################################################################
class LpPiecewise2(LpBase):
    """
    Piecewise Linear Program from Talluri & Van Ryzin (3.8).

    For each product j and each unit of capacity d = 1, ..., M_j, define
    variable z_{j,d} in [0, 1] representing the d-th unit of capacity
    allocated to product j.  The objective coefficient is p_j * P(D_j >= d),
    the marginal expected revenue of that unit.  Because P(D_j >= d) is
    non-increasing in d, the LP automatically fills lower-indexed (higher
    probability) units first.

    For continuous normal demand:
        P(D_j >= d) = 1 - Phi((d - mu_j) / sigma_j)

    The resulting LP is:

        max   sum_j  p_j * sum_{d=1}^{M_j}  P(D_j >= d) * z_{j,d}
        s.t.  sum_{j uses leg l}  sum_d  z_{j,d}  <=  cap_l - sold_l
              0 <= z_{j,d} <= 1

    M_j is the minimum remaining capacity across all legs used by product j,
    which is the tightest feasible upper bound on allocations to that product.

    The dual of each leg capacity constraint is the displacement cost (bid
    price), written back to leg.displacement.

    Parameters
    ----------
    sim : SimulationEngine
    carrier : str
    epsilon : float
        Treat demand mean/sigma below this as effectively zero.
    _debug : bool
    """

    __slots__ = (
        "carrier",
        "solver",
        "objective",
        "lp_vars",
        "constraints",
        "epsilon",
        "debug",
    )

    def __init__(
        self,
        carrier: str,
        epsilon: float = 0.01,
        _debug: bool = False,
    ):
        self.carrier = carrier
        self.epsilon = epsilon
        self.debug = _debug

    # Build (or rebuild) the LP for the current simulation state
    def initialize(self, eng: SimulationEngine):
        """
        Construct the LP from current path-class forecasts and remaining
        leg capacities.  Called once per DCP.

        Implements (3.8):
          - Variable z_{j,d} in [0,1] for each product j, unit d = 1..M_j
          - Objective coefficient: p_j * P(D_j >= d)
          - Capacity constraint per leg: sum of all z_{j,d} for products
            on that leg <= remaining capacity

        Requires sim.build_connections() to have been called so that
        path.num_legs() and path.get_leg_fltno() are populated.
        """
        self.solver = pywraplp.Solver.CreateSolver("GLOP")
        self.objective = self.solver.Objective()
        self.objective.SetMaximization()

        tmp = DynamicProgram(eng, self.carrier)
        tmp.initialize()  # This step sets up the PcPtr structures, will refactor later to make this a separate call

        self.lp_vars = {}

        if self.debug:
            print(f"********** Sample={eng.sample}  PNLP setup — carrier={self.carrier} **********")

        # Collect remaining capacity per leg
        leg_remaining: dict[int, int] = {}
        for leg in eng.legs:
            if leg.carrier_name != self.carrier:
                continue
            leg_remaining[leg.flt_no] = int(leg.capacity - leg.sold)

        _std = NormalDist()

        # Decision variables: z_{pc_id, d}  for d = 1, ..., M_j
        #
        # M_j = min remaining capacity over all legs on this path, or the
        # tightest upper bound on how many seats this product can consume.
        for path in eng.paths.set_filters(carrier=self.carrier):
            legs_on_path = [path.get_leg_fltno(i) for i in range(path.num_legs())]
            M_j = int(
                min(
                    (leg_remaining[f] for f in legs_on_path if f in leg_remaining),
                    default=0,
                )
            )

            for pc in path.pathclasses:
                fare = pc.decision_fare
                mu = pc.fcst_mean
                sigma = pc.fcst_std_dev

                if self.debug:
                    print(f"  PathClass {pc.identifier}: fare={fare:.2f}, mu={mu:.3f}, sigma={sigma:.3f}, M_j={M_j}")

                for d in range(1, M_j + 1):
                    # P(D_j >= d): probability that demand meets or exceeds
                    # this unit, or the marginal value of accepting one more seat
                    if mu < self.epsilon or sigma < self.epsilon:
                        # prob = 1.0 if d <= mu else 0.0
                        prob = 0.0
                    else:
                        prob = 1.0 - _std.cdf((d - mu) / sigma)

                    name = f"{pc.identifier}-{d}"
                    var = self.solver.NumVar(0.0, 1.0, name)
                    self.lp_vars[name] = var
                    self.objective.SetCoefficient(var, fare * prob)

                    if self.debug:
                        print(f"    d={d}: P(D>={d})={prob:.4f}  coeff={fare * prob:.4f}")

        # Capacity constraints: one per leg
        #   sum_{j uses leg l}  sum_{d=1}^{M_j}  z_{j,d}  <--  remaining_l
        self.constraints = {}

        for leg in eng.legs:
            if leg.carrier_name != self.carrier:
                continue
            ct = self.solver.Constraint(0.0, float(leg_remaining[leg.flt_no]), f"Leg-{leg.flt_no}")
            self.constraints[leg.flt_no] = ct

        if self.debug:
            print("  CAPACITY CONSTRAINTS")

        for path in eng.paths.set_filters(carrier=self.carrier):
            legs_on_path = [path.get_leg_fltno(i) for i in range(path.num_legs())]
            M_j = int(
                min(
                    (leg_remaining[f] for f in legs_on_path if f in leg_remaining),
                    default=0,
                )
            )
            for pc in path.pathclasses:
                for d in range(1, M_j + 1):
                    name = f"{pc.identifier}-{d}"
                    var = self.lp_vars[name]
                    for flt_no in legs_on_path:
                        if flt_no in self.constraints:
                            self.constraints[flt_no].SetCoefficient(var, 1.0)
                            if self.debug:
                                print(f"    Leg {flt_no}: var={name}")

    def update(self, eng: SimulationEngine):
        self.initialize(eng)

    # Solve and write displacement costs (bid prices) back to the engine
    def solve(self, eng: SimulationEngine, debug: bool = False):
        """
        Solve the LP and write dual values (displacement costs / bid prices)
        back to each leg.
        """
        status = self.solver.Solve()

        if self.test_status(status):
            for leg in eng.legs:
                if leg.carrier_name != self.carrier:
                    continue
                # Dual of the capacity constraint = displacement cost.
                # Clamp to zero; tiny negatives can arise from numerical noise.
                dual = max(self.constraints[leg.flt_no].dual_value(), 0.0)
                leg.displacement = dual
                leg.bid_price = dual
                if debug:
                    print(f"  Leg {leg.flt_no}: displacement={dual:.4f}")


# ##########################################################################################
class SSBLP(LpBase):
    """Stochastic Sales-Based LP, following [Ratliff 2025]"""

    pass
