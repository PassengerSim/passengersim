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

from collections.abc import Callable

import pandas as pd

from .carriers import aggregate_carriers, extract_carriers
from .generic import SimulationTableItem, _GenericSimulationTables
from .legs import aggregate_legs, extract_legs
from .segmentation_by_timeframe import (
    aggregate_segmentation_by_timeframe,
    extract_segmentation_by_timeframe,
)


class SimulationTables(_GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of _GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    segmentation_by_timeframe: pd.DataFrame
    legs: pd.DataFrame
    carriers: pd.DataFrame

    fig_segmentation_by_timeframe: Callable
