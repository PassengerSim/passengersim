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

from . import (
    bid_price_history,
    carriers,
    demand_to_come,
    demands,
    displacement_history,
    edgar,
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
