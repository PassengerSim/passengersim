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

from collections.abc import Sized
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
    segmentation_detail,
)
from .generic import GenericSimulationTables


def _describe_type(obj):
    """Return a string describing the type of an object."""
    if isinstance(obj, pd.DataFrame):
        typename = f"{len(obj)} row DataFrame"
    else:
        try:
            typename = str(type(obj).__name__)
        except AttributeError:
            typename = str(type(obj))
        if isinstance(obj, Sized):
            typename = f"{len(obj)} item {typename}"
    return typename


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
            table_keys.append(f"* {k} ({_describe_type(v)})")
        for k, v in self._callback_data.items():
            table_keys.append(f"* callback_data.{k} ({_describe_type(v)})")
        if self._file_store is not None:
            for k in self._file_store:
                if not k.startswith("_") and k not in self._data:
                    table_keys.append(f"* {k} (available in file storage)")
                if k == "_callback_data_" and not self._callback_data:
                    table_keys.append("* callback_data (available in file storage)")
        if table_keys:
            table_info = " " + "\n ".join(table_keys)
        else:
            table_info = " (no tables)"
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

    def dashboard(self):
        """Return a dashboard object for this SimulationTables instance."""
        from .dashboard import default_dashboard

        return default_dashboard(self)
