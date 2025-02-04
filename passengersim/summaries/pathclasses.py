from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from passengersim.reporting import report_figure

from .fare_class_mix import _fig_fare_class_mix
from .generic import GenericSimulationTables, SimulationTableItem
from .tools import aggregate_by_summing_dataframe

if TYPE_CHECKING:
    from passengersim import Simulation


def extract_pathclasses(sim: Simulation) -> pd.DataFrame | None:
    """Extract leg-level summary data from a Simulation."""
    pc_data = []
    for pth in sim.sim.paths:
        for pc in pth.pathclasses:
            this_pc = {
                "path_id": pth.path_id,
                "booking_class": pc.booking_class,
                "carrier": pth.carrier_name,
                "orig": pth.orig,
                "dest": pth.dest,
                "gt_sold": pc.gt_sold,
                "gt_sold_priceable": pc.gt_sold_priceable,
                "gt_revenue": pc.gt_revenue,
            }
            gt_sold_by_segment = pc.gt_sold_by_segment
            if gt_sold_by_segment:
                for seg, sold in gt_sold_by_segment.items():
                    this_pc[f"gt_sold_by_segment_{seg}"] = sold
            pc_data.append(this_pc)
    if len(pc_data) == 0:
        return None
    df = pd.DataFrame(pc_data).set_index(["path_id", "booking_class"]).sort_index()
    df = df.fillna(0)
    return df


class SimTabPathClasses(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    pathclasses: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_by_summing_dataframe(
            "pathclasses", ["carrier", "orig", "dest"]
        ),
        extraction_func=extract_pathclasses,
        doc="Path-Class summary data.",
    )

    @report_figure
    def fig_od_fare_class_mix(
        self, orig: str, dest: str, *, raw_df=False, label_threshold=0.06
    ) -> pd.DataFrame:
        df = (
            self.pathclasses[
                (self.pathclasses["orig"] == orig) & (self.pathclasses["dest"] == dest)
            ]
            .groupby(["carrier", "booking_class"])["gt_sold"]
            .sum()
            / self.n_total_samples
        )
        df = df.rename("avg_sold")
        df = df.reset_index()[["carrier", "booking_class", "avg_sold"]]

        if raw_df:
            return df

        return _fig_fare_class_mix(
            df,
            label_threshold=label_threshold,
            title=f"Fare Class Mix for {orig} to {dest}",
        )
