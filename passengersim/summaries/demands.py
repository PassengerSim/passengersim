from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .generic import GenericSimulationTables, SimulationTableItem
from .tools import aggregate_by_summing_dataframe

if TYPE_CHECKING:
    from passengersim import Simulation


def extract_demands(sim: Simulation) -> pd.DataFrame | None:
    """Extract demand-level summary data from a Simulation."""
    dmd_data = []
    for dmd in sim.sim.demands:
        dmd_data.append(
            {
                "orig": dmd.orig,
                "dest": dmd.dest,
                "segment": dmd.segment,
                "base_demand": dmd.base_demand,
                "reference_fare": dmd.reference_fare,
                "distance": dmd.distance,
                "gt_demand": dmd.gt_demand,
                "gt_revenue": dmd.gt_revenue,
                "gt_sold": dmd.gt_sold,
                "gt_eliminated_no_offers": dmd.gt_eliminated_no_offers,
                "gt_eliminated_chose_nothing": dmd.gt_eliminated_chose_nothing,
                "gt_eliminated_wtp": dmd.gt_eliminated_wtp,
            }
        )
    if len(dmd_data) == 0:
        return None
    return pd.DataFrame(dmd_data).set_index(["orig", "dest", "segment"])


class SimTabDemands(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    demands: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_by_summing_dataframe(
            "demands",
            extra_idxs=["base_demand", "reference_fare", "distance"],
        ),  # don't sum extra_idxs, they should be identical over trials
        extraction_func=extract_demands,
        doc="Demand-level summary data.",
    )
