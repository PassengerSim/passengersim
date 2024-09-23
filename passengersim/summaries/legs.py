from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .generic import SimulationTableItem, _GenericSimulationTables
from .tools import aggregate_by_summing_dataframe

if TYPE_CHECKING:
    from passengersim import Simulation


def extract_legs(sim: Simulation) -> pd.DataFrame:
    """Extract leg-level summary data from a Simulation."""
    leg_data = []
    for leg in sim.sim.legs:
        leg_data.append(
            {
                "leg_id": leg.leg_id,
                "carrier": leg.carrier.name,
                "flt_no": leg.flt_no,
                "orig": leg.orig,
                "dest": leg.dest,
                "gt_sold": leg.gt_sold,
                "gt_capacity": leg.gt_capacity,
                "gt_sold_local": leg.gt_sold_local,
                "gt_revenue": leg.gt_revenue,
                "distance": leg.distance,
            }
        )
    return pd.DataFrame(leg_data).set_index("leg_id")


class SimTabLegs(_GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of _GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    legs: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_by_summing_dataframe(
            "legs", ["carrier", "flt_no", "orig", "dest", "distance"]
        ),
        extraction_func=extract_legs,
        computed_fields={
            "avg_load_factor": "100.0 * gt_sold / gt_capacity",
            "avg_local": "100.0 * gt_sold_local / gt_sold",
        },
        doc="Leg-level summary data.",
    )
