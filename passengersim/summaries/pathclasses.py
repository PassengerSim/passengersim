from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from passengersim.reporting import report_figure

from .fare_class_mix import _fig_fare_class_mix
from .generic import GenericSimulationTables, SimulationTableItem
from .tools import aggregate_by_summing_dataframe

if TYPE_CHECKING:
    import altair as alt

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
            gt_revenue_by_segment = pc.gt_revenue_by_segment
            if gt_revenue_by_segment:
                for seg, rev in gt_revenue_by_segment.items():
                    this_pc[f"gt_revenue_by_segment_{seg}"] = rev
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
        aggregation_func=aggregate_by_summing_dataframe("pathclasses", ["carrier", "orig", "dest"]),
        extraction_func=extract_pathclasses,
        doc="Path-Class summary data.",
    )

    @report_figure
    def fig_od_fare_class_mix(
        self, orig: str, dest: str, *, raw_df=False, also_df: bool = False, label_threshold=0.06
    ) -> alt.Chart | pd.DataFrame | tuple[alt.Chart, pd.DataFrame]:
        """
        Plot the fare class mix data for a specific origin-destination pair.

        Parameters
        ----------
        orig, dest : str
            The origin and destination airport codes.
        raw_df : bool, optional
            If True, return the raw dataframe instead of the figure.
        also_df : bool, optional
            If True, return the dataframe as well as the figure.
        label_threshold : float, optional
            The threshold for displaying labels on the bars. Default is 0.06.

        Returns
        -------
        alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
            The fare class mix figure or dataframe.
        """
        df = (
            self.pathclasses[(self.pathclasses["orig"] == orig) & (self.pathclasses["dest"] == dest)]
            .groupby(["carrier", "booking_class"])["gt_sold"]
            .sum()
            / self.n_total_samples
        )
        df = df.rename("avg_sold")
        df = df.reset_index()[["carrier", "booking_class", "avg_sold"]]

        if raw_df:
            return df

        fig = _fig_fare_class_mix(
            df,
            label_threshold=label_threshold,
            title=f"Fare Class Mix for {orig} to {dest}",
        )
        if also_df:
            return fig, df
        return fig

    @property
    def market_segmentation(self) -> pd.DataFrame:
        """Computed DataFrame with market segmentation data."""
        if "market_segmentation" not in self._data:
            if self.pathclasses is None:
                raise ValueError("pathclasses not found, cannot compute market_segmentation")
            df = self.pathclasses.groupby(["carrier", "orig", "dest"]).sum() / self.n_total_samples
            df.columns = df.columns.str.removeprefix("gt_")
            d = {}
            for metric in ["sold", "revenue"]:
                dfm = df.filter(like=f"{metric}_by_segment")
                dfm.columns = dfm.columns.str.replace(f"{metric}_by_segment_", "", regex=False)
                dfm.columns.name = "segment"
                d[metric] = dfm.stack()
            self._data["market_segmentation"] = pd.concat(d, axis=1)
        return self._data["market_segmentation"]
