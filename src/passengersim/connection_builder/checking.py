from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from passengersim import Simulation
    from passengersim.core import SimulationEngine


def check_connections(sim: Simulation | SimulationEngine, min_paths_per_market: int = 1):
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
