from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .generic import SimulationTableItem, _GenericSimulationTables
from .tools import aggregate_by_summing_dataframe

if TYPE_CHECKING:
    from passengersim import Simulation


def extract_paths(sim: Simulation) -> pd.DataFrame:
    """Extract path-level summary data from a Simulation."""
    path_data = []
    for pth in sim.sim.paths:
        path_data.append(
            {
                "path_id": pth.path_id,
                "carrier": pth.carrier,
                "orig": pth.orig,
                "dest": pth.dest,
                "gt_sold": pth.gt_sold,
                "gt_sold_priceable": pth.gt_sold_priceable,
                "gt_revenue": pth.gt_revenue,
                "hhi": pth.hhi,
                "max_hhi": pth.max_hhi,
                "minimum_connect_time": pth.minimum_connect_time,
                "num_legs": pth.num_legs(),
                "path_quality_index": pth.path_quality_index,
            }
        )
    return pd.DataFrame(path_data).set_index("path_id")


class SimTabPaths(_GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of _GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    paths: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_by_summing_dataframe(
            "paths",
            [
                "carrier",
                "orig",
                "dest",
                "num_legs",
                "hhi",
                "max_hhi",
                "path_quality_index",
                "minimum_connect_time",
            ],
        ),
        extraction_func=extract_paths,
        doc="Path-level summary data.",
    )
