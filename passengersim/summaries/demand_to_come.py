from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from passengersim.utils.nested_dict import from_nested_dict

from .generic import GenericSimulationTables, SimulationTableItem
from .tools import combine_sigmas

if TYPE_CHECKING:
    from passengersim import Simulation

    from . import SimulationTables


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


def aggregate_demand_to_come(
    summaries: list[SimulationTables],
) -> pd.DataFrame | None:
    frames = []
    for s in summaries:
        frame = getattr(s, "_raw_demand_to_come", None)
        if frame is not None:
            frames.append((frame, s.n_total_samples))
    while len(frames) > 1:
        df, df_n = frames[0]
        other, other_n = frames.pop(1)
        df["stdev_future_demand"] = np.sqrt(
            combine_sigmas(
                df["stdev_future_demand"],
                other["stdev_future_demand"],
                df["mean_future_demand"],
                other["mean_future_demand"],
                df_n,
                other_n,
                ddof=1,
            )
        )
        df["mean_future_demand"] = (
            df["mean_future_demand"] * df_n + other["mean_future_demand"] * other_n
        ) / (df_n + other_n)
        frames[0] = (df, df_n + other_n)
    if frames:
        return frames[0][0]
    return None


class SimTabDemandToCome(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    demand_to_come: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_demand_to_come,
        extraction_func=extract_demand_to_come,
        doc="Demand-to-come summary data.",
    )