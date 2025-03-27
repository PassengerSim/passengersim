from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .generic import GenericSimulationTables, SimulationTableItem
from .tools import aggregate_by_summing_dataframe

if TYPE_CHECKING:
    from passengersim import Simulation


def extract_paths(sim: Simulation) -> pd.DataFrame | None:
    """Extract path-level summary data from a Simulation."""
    path_data = []
    for pth in sim.sim.paths:
        path_data.append(
            {
                "path_id": pth.path_id,
                "carrier": pth.carrier_name,
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
    if len(path_data) == 0:
        return None
    return pd.DataFrame(path_data).set_index("path_id")


class SimTabPaths(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of GenericSimulationTables, which is defined in
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

    def path_identifier(self, path_id: int) -> str:
        """
        Get a human-readable identifying string for a path.

        Parameters
        ----------
        path_id : int
            The path_id to look up.

        Returns
        -------
        str
        """

        def q(attribute):
            return self.paths.loc[path_id, attribute]

        s = f"Path {path_id}: {q('orig')}~{q('dest')},"
        leg_ids = self.path_legs.query("path_id == @path_id")["leg_id"]
        legs = self.legs.query("leg_id in @leg_ids")
        for i in leg_ids:
            s += (
                f" ({legs.loc[i, 'carrier']}:{legs.loc[i, 'flt_no']}"
                f" {legs.loc[i, 'orig']}-{legs.loc[i, 'dest']})"
            )
        return s
