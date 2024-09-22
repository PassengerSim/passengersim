from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from passengersim.utils.nested_dict import from_nested_dict

from .generic import SimulationTableItem, _GenericSimulationTables
from .tools import aggregate_by_concat_dataframe

if TYPE_CHECKING:
    from passengersim import Simulation


def extract_demand_to_come(sim: Simulation) -> pd.DataFrame:
    """Extract demand-to-come summary data from a Simulation."""
    eng = sim.sim
    raw = eng.summary_demand_to_come()
    df = (
        from_nested_dict(raw, ["segment", "days_prior", "metric"])
        .sort_index(ascending=[True, False])
        .rename(columns={"mean": "mean_future_demand", "stdev": "stdev_future_demand"})
    )
    return df


class SimTabDemandToCome(_GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of _GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    demand_to_come: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_by_concat_dataframe("demand_to_come"),
        extraction_func=extract_demand_to_come,
        doc="Demand-to-come summary data.",
    )
