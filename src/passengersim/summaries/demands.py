from __future__ import annotations

from typing import TYPE_CHECKING

import altair as alt
import pandas as pd

from passengersim.reporting import report_figure

from .generic import GenericSimulationTables, SimulationTableItem
from .tools import aggregate_by_summing_dataframe

if TYPE_CHECKING:
    from passengersim import Simulation


def extract_demands(sim: Simulation) -> pd.DataFrame | None:
    """Extract demand-level summary data from a Simulation."""
    dmd_data = []
    for dmd in sim.eng.demands:
        dmd_data.append(
            {
                "orig": dmd.orig,
                "dest": dmd.dest,
                "segment": dmd.segment,
                "base_demand": dmd.base_demand,
                "reference_price": dmd.reference_price,
                "distance": dmd.distance,
                "gt_demand": dmd.gt_demand,
                "gt_revenue": dmd.gt_revenue,
                "gt_sold": dmd.gt_sold,
                "gt_eliminated_no_offers": dmd.gt_eliminated_no_offers,
                "gt_eliminated_chose_nothing": dmd.gt_eliminated_chose_nothing,
                "gt_eliminated_wtp": dmd.gt_eliminated_wtp,
            }
        )
    if len(dmd_data) == 0:
        return None
    return pd.DataFrame(dmd_data).set_index(["orig", "dest", "segment"])


class SimTabDemands(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    demands: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_by_summing_dataframe(
            "demands",
            extra_idxs=["base_demand", "reference_price", "distance"],
        ),  # don't sum extra_idxs, they should be identical over trials
        extraction_func=extract_demands,
        doc="Demand-level summary data.",
    )

    @report_figure
    def fig_demand_segmentation_distribution(
        self,
        x: str | None = None,
        y: str | None = None,
        *,
        raw_df: bool = False,
        also_df: bool = False,
    ) -> alt.Chart | pd.DataFrame | tuple[alt.Chart, pd.DataFrame]:
        """
        Create a scatter plot showing the distribution of demands by segment.

        Parameters
        ----------
        x : str, optional
            The column to use for the x-axis. If not provided, the first segment column
            will be used.
        y : str, optional
            The column to use for the y-axis. If not provided, the second segment column
            will be used if there are two segments, otherwise 'total' will be used.
        raw_df : bool, default False
            If True, return the raw DataFrame used to create the plot instead of the plot itself
        also_df : bool, default False
            If True, return a tuple of (plot, DataFrame) instead of just the plot

        Returns
        -------
        alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
            The scatter plot, the raw DataFrame, or both, depending on the parameters.
        """
        df = self.demands.pivot_table(
            index=["orig", "dest"], columns="segment", values="base_demand", aggfunc="sum"
        ).fillna(0)

        # determine x and y if not provided
        if x is None:
            x = df.columns[0]
        if y is None:
            y = df.columns[1] if len(df.columns) == 2 else "total"

        # compute total and ratios
        total = df.sum(axis=1)
        ratios = df.div(total, axis=0).add_suffix("_share")
        df["total"] = total
        df = pd.concat([df, ratios], axis=1)
        if raw_df:
            return df
        df = df.reset_index()
        tooltips = []
        for col in df.columns:
            if col in ["orig", "dest"]:
                tooltips.append(col)
            elif col.endswith("_share"):
                tooltips.append(alt.Tooltip(col, format=".2%"))
            else:
                tooltips.append(alt.Tooltip(col, format=".2f"))
        chart = (
            alt.Chart(df)
            .mark_point()
            .encode(x=x, y=y, tooltip=tooltips)
            .properties(title="Demand Segmentation Distribution")
        )
        if also_df:
            return chart.interactive(), df
        return chart.interactive()
