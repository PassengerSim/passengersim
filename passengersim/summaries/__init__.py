from collections.abc import Callable

import pandas as pd

from .generic import SimulationTableItem, _GenericSimulationTables
from .legs import aggregate_legs, extract_legs
from .segmentation_by_timeframe import (
    aggregate_segmentation_by_timeframe,
    extract_segmentation_by_timeframe,
)


class SimulationTables(_GenericSimulationTables):
    segmentation_by_timeframe: pd.DataFrame
    legs: pd.DataFrame

    fig_segmentation_by_timeframe: Callable
