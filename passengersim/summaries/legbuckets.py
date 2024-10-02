from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .generic import GenericSimulationTables, SimulationTableItem
from .tools import aggregate_by_summing_dataframe

if TYPE_CHECKING:
    from passengersim import Simulation


def extract_legbuckets(sim: Simulation) -> pd.DataFrame | None:
    """Extract leg-bucket-level summary data from a Simulation."""
    bkt_data = []
    for leg in sim.sim.legs:
        for bkt in leg.buckets:
            bkt_data.append(
                {
                    "leg_id": leg.leg_id,
                    "booking_class": bkt.name,
                    "carrier": leg.carrier.name,
                    "gt_sold": bkt.gt_sold,
                    "gt_revenue": bkt.gt_revenue,
                }
            )
    if len(bkt_data) == 0:
        return None
    return pd.DataFrame(bkt_data).set_index(["leg_id", "booking_class"]).sort_index()


class SimTabLegBuckets(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    legbuckets: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_by_summing_dataframe("legbuckets", ["carrier"]),
        extraction_func=extract_legbuckets,
        doc="Leg-Bucket summary data.",
    )
