from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from passengersim import Simulation
    from passengersim.core import SimulationEngine


def check_connections(sim: Simulation | SimulationEngine, min_paths_per_market: int = 1) -> None:
    """Validate that the connection structure of a simulation is well-formed.

    Checks that every demand has a corresponding market, and that every market
    has at least the required minimum number of itinerary paths.  Collects all
    detected problems before raising so that every issue is reported at once.

    Parameters
    ----------
    sim : Simulation or SimulationEngine
        The simulation (or underlying engine) whose connections are to be
        checked.
    min_paths_per_market : int, optional
        The minimum number of paths that each market must contain.
        Defaults to 1.

    Raises
    ------
    ValueError
        If any problems are found, a ``ValueError`` is raised that lists every
        detected problem.  Problems include markets that are missing for a
        demand, markets with no paths, and markets whose path count is below
        *min_paths_per_market*.
    """
    problems = []

    # Check that a market has been created for each demand
    for d in sim.demands:
        if d.market_identifier not in sim.markets:
            problems.append(f"Market {d.market_identifier} not found for demand {d.identifier}")

    # check that every market has at least some minimum number of paths
    for mkt_id, mkt in sim.markets.items():
        if len(mkt.paths) == 0:
            problems.append(f"Market {mkt_id} has no paths")
        elif len(mkt.paths) < min_paths_per_market:
            problems.append(f"Market {mkt_id} has only {len(mkt.paths)} path(s) [{', '.join(map(str, mkt.paths))}]")

    if problems:
        raise ValueError(f"{len(problems)} problems found in connections:\n- " + "\n- ".join(problems))
