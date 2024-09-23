from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .generic import SimulationTableItem, _GenericSimulationTables
from .tools import aggregate_by_summing_dataframe

if TYPE_CHECKING:
    from passengersim import Simulation


def extract_pathclasses(sim: Simulation) -> pd.DataFrame:
    """Extract leg-level summary data from a Simulation."""
    pc_data = []
    for pth in sim.sim.paths:
        for pc in pth.pathclasses:
            pc_data.append(
                {
                    "path_id": pth.path_id,
                    "booking_class": pc.booking_class,
                    "carrier": pth.carrier,
                    "orig": pth.orig,
                    "dest": pth.dest,
                    "gt_sold": pc.gt_sold,
                    "gt_sold_priceable": pc.gt_sold_priceable,
                    "gt_revenue": pc.gt_revenue,
                }
            )
    return pd.DataFrame(pc_data).set_index(["path_id", "booking_class"]).sort_index()


class SimTabPathClasses(_GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of _GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    pathclasses: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_by_summing_dataframe(
            "pathclasses", ["carrier", "orig", "dest"]
        ),
        extraction_func=extract_pathclasses,
        doc="Path-Class summary data.",
    )
