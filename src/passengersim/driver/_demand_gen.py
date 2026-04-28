from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from passengersim.config.simulation_controls import SimulationSettings
    from passengersim.core import SimulationEngine
    from passengersim.iterators.demand import DemandIterator


def generate_sample_demands(
    engine: SimulationEngine,
    simulation_controls: SimulationSettings,
    *,
    allocate: bool = True,
) -> None:
    """Generate scenario demand values for each demand record.

    This applies the configured randomness factors to each base demand. The
    resulting scenario demand is written back to each demand object.

    When ``allocate`` is true, the rounded scenario demand is also distributed
    across timeframes by calling the engine's configured allocation routine.

    Parameters
    ----------
    engine : SimulationEngine
        Simulation engine that supplies demand records, random-number sources,
        allocation routines, and firehose logging.
    simulation_controls : SimulationSettings
        Simulation settings that define the demand variability and timeframe
        allocation behavior.
    allocate : bool, default True
        Whether to allocate the generated scenario demand into timeframes
        immediately after generation.

    Raises
    ------
    ValueError
        Raised if the timeframe allocator produces a number of events that does
        not match the rounded scenario demand.
    """
    random_generator = engine.random_generator
    system_rn = random_generator.get_normal()
    end_time = int(engine.base_time)

    # this stores a random number per passenger segment and per market,
    # which is re-used across those, inducing correlation.
    segment_random_cache = {}
    market_random_cache = {}

    def get_or_make_random(grouping, key):
        """Return a cached normal draw for ``key``, creating it on first use."""
        if key not in grouping:
            grouping[key] = random_generator.get_normal()
        return grouping[key]

    for dmd in engine.demands:
        base = dmd.base_demand

        if dmd.deterministic:
            # Deterministic demand, no randomness
            dmd.scenario_demand = base
        else:
            # Get the random numbers we're going to use to perturb demand
            mrn = get_or_make_random(market_random_cache, (dmd.orig, dmd.dest))
            if simulation_controls.segment_k_factor:
                srn = get_or_make_random(segment_random_cache, dmd.segment)
            else:
                srn = 0
            if simulation_controls.simple_cv100 > 0.0:
                sigma = simulation_controls.simple_cv100 * sqrt(base) * 10.0
                urn = random_generator.get_normal() * sigma
            elif simulation_controls.simple_k_factor:
                urn = random_generator.get_normal() * simulation_controls.simple_k_factor
            else:
                urn = 0

            mu = base * (
                1.0
                + system_rn * simulation_controls.sys_k_factor
                + mrn * simulation_controls.mkt_k_factor
                + srn * simulation_controls.segment_k_factor
                + urn
            )
            mu = max(mu, 0.0)
            sigma = sqrt(mu * simulation_controls.tot_z_factor)  # Correct?
            n = mu + sigma * random_generator.get_normal()
            dmd.scenario_demand = max(n, 0)

            engine.write_to_firehose(
                "demand",
                "DMD,{engine.sample},{dmd.orig},{dmd.dest},"
                "{dmd.segment},{dmd.base_demand},"
                "{mu:.3f},{sigma:.3f},{n:.0f}\n",
                engine=engine,
                dmd=dmd,
                mu=mu,
                sigma=sigma,
                n=n,
            )

        if allocate:
            # split total sample demand up over timeframes and add it to the simulation
            num_pax = int(dmd.scenario_demand + 0.5)  # rounding
            if simulation_controls.timeframe_demand_allocation == "pods":
                num_events_by_tf = engine.allocate_demand_to_tf_pods(
                    dmd, num_pax, simulation_controls.tf_k_factor, end_time
                )
            else:
                num_events_by_tf = engine.allocate_demand_to_tf(dmd, num_pax, simulation_controls.tf_k_factor, end_time)
            num_events = sum(num_events_by_tf)
            if num_events != round(num_pax):
                raise ValueError(
                    f"inconsistent result in demand allocation, "
                    f"expected to allocate {num_pax} customers, "
                    f"actually allocated {num_events} customers"
                )


def allocate_sample_demands(
    engine: SimulationEngine,
    simulation_controls: SimulationSettings,
) -> None:
    """Allocate previously generated scenario demand across timeframes.

    This function uses each demand object's existing ``scenario_demand`` value,
    rounds it to a passenger count, and pushes that count through the standard
    or PODS timeframe allocator configured in the simulation settings.

    Parameters
    ----------
    engine
        Simulation engine that supplies demand records and allocation methods.
    simulation_controls
        Simulation settings that define which timeframe allocator to use and
        the timeframe variability factor.

    Raises
    ------
    ValueError
        Raised if the allocator returns a number of events that does not match
        the rounded scenario demand.
    """
    end_time = int(engine.base_time)
    for dmd in engine.demands:
        # split total sample demand up over timeframes and add it to the simulation
        num_pax = int(dmd.scenario_demand + 0.5)  # rounding
        if simulation_controls.timeframe_demand_allocation == "pods":
            num_events_by_tf = engine.allocate_demand_to_tf_pods(
                dmd, num_pax, simulation_controls.tf_k_factor, end_time
            )
        else:
            num_events_by_tf = engine.allocate_demand_to_tf(dmd, num_pax, simulation_controls.tf_k_factor, end_time)
        num_events = sum(num_events_by_tf)
        if num_events != round(num_pax):
            raise ValueError(
                f"inconsistent result in demand allocation, "
                f"expected to allocate {num_pax} customers, "
                f"actually allocated {num_events} customers"
            )


def get_total_sample_demands(demands: DemandIterator) -> dict[tuple[str, str, str], float]:
    """Return a snapshot of scenario demand keyed by market and segment.

    Parameters
    ----------
    demands
        Iterator of demand objects whose ``scenario_demand`` values should be
        collected.

    Returns
    -------
    dict[tuple[str, str, str], float]
        Mapping from ``(orig, dest, segment)`` to each demand's scenario value.
    """
    result = {}
    for dmd in demands:
        result[(dmd.orig, dmd.dest, dmd.segment)] = dmd.scenario_demand
    return result


def get_base_demands(demands: DemandIterator) -> dict[tuple[str, str, str], float]:
    """Return a snapshot of base demand keyed by market and segment.

    Parameters
    ----------
    demands
        Iterator of demand objects whose ``base_demand`` values should be
        collected.

    Returns
    -------
    dict[tuple[str, str, str], float]
        Mapping from ``(orig, dest, segment)`` to each demand's base value.
    """
    result = {}
    for dmd in demands:
        result[(dmd.orig, dmd.dest, dmd.segment)] = dmd.base_demand
    return result
