"""
This subpackage is for defining the SimulationTables class and its items.

The SimulationTables class is a container for summary tables and figures
that are extracted from a Simulation object after it has been run.

Each item in the SimulationTables class is defined by a SimulationTableItem
instance, which specifies how to extract and aggregate the summary data.
Each of these is defined in a separate module in this subpackage.  Within
each module, the SimulationTableItem instance is created using the
`.generic.SimulationTable_add_item()` function.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from . import (
    bid_price_history,
    carriers,
    demand_to_come,
    demands,
    displacement_history,
    fare_class_mix,
    forecasts,
    legbuckets,
    legs,
    local_and_flow_yields,
    pathclasses,
    pathlegs,
    paths,
    segmentation_by_timeframe,
)
from .generic import GenericSimulationTables


class SimulationTables(*GenericSimulationTables.subclasses()):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    __final__ = True

    def __repr__(self) -> str:
        """Includes information about what data is stored in this summary."""
        table_keys = []
        for k, v in self._data.items():
            if isinstance(v, pd.DataFrame):
                table_keys.append(f"* {k} ({len(v)} row DataFrame)")
            elif v is not None:
                try:
                    typename = type(v).__name__
                except AttributeError:
                    typename = str(type(v))
                table_keys.append(f"* {k} ({typename})")
        table_info = " " + "\n ".join(table_keys)
        try:
            timestamp = self.metadata("time.created")
        except KeyError:
            time_created = ""
        else:
            date_string = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
            time_created = f" created on {date_string}"
        return (
            f"<passengersim.summaries.SimulationTables{time_created}>\n"
            f"{table_info}\n<*>"
        )
