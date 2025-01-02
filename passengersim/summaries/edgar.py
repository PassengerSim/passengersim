#
# A simple implementation of forecast errors, using the EDGAR approach
#
# AlanW, December 2024
# (c) PassengerSim LLC
#

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

# from passengersim.reporting import report_figure
from passengersim.database import common_queries

from .generic import GenericSimulationTables, DatabaseTableItem
from .tools import aggregate_by_concat_dataframe

if TYPE_CHECKING:
    from passengersim import Simulation


def extract_edgar(sim: Simulation) -> pd.DataFrame | None:
    df = pd.DataFrame()
    return df


class SimTabEdgar(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of _Generic
    """

    edgar: pd.DataFrame = DatabaseTableItem(
        aggregation_func=aggregate_by_concat_dataframe("edgar"),
        query_func=common_queries.edgar,
#        extraction_func=extract_edgar,
        doc="Detailed forecast error data.",
    )

