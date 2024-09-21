from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .generic import SimulationTable_add_item, SimulationTables

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


def aggregate_legs(summaries: list[SimulationTables]) -> pd.DataFrame | None:
    """Aggregate leg-level summaries."""
    table_sum = []
    for s in summaries:
        frame = s._raw_legs
        if frame is not None:
            table_sum.append(
                frame.set_index(
                    ["carrier", "flt_no", "orig", "dest", "distance"], append=True
                )
            )
    while len(table_sum) > 1:
        table_sum[0] = table_sum[0].add(table_sum.pop(1), fill_value=0)
    if table_sum:
        return table_sum[0].reset_index(
            ["carrier", "flt_no", "orig", "dest", "distance"]
        )
    return None


SimulationTable_add_item(
    "legs",
    aggregation_func=aggregate_legs,
    extraction_func=extract_legs,
    computed_fields={
        "avg_load_factor": "100.0 * gt_sold / gt_capacity",
        "avg_local": "100.0 * gt_sold_local / gt_sold",
    },
    doc="Leg-level summary data.",
)
