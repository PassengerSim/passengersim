from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from passengersim.database import common_queries

from .generic import DatabaseTableItem, GenericSimulationTables
from .tools import aggregate_by_averaging_dataframe

if TYPE_CHECKING:
    pass


class SimTabLocalAndFlowYields(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    local_and_flow_yields: pd.DataFrame = DatabaseTableItem(
        aggregation_func=aggregate_by_averaging_dataframe(
            "local_and_flow_yields",
            extra_idxs=["leg_id", "carrier", "orig", "dest", "capacity", "distance"],
        ),
        query_func=common_queries.local_and_flow_yields,
        doc="Local and flow yields.",
    )
