from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from passengersim.reporting import report_figure

from .generic import GenericSimulationTables, SimulationTableItem
from .tools import aggregate_by_concat_dataframe

if TYPE_CHECKING:
    from passengersim import Simulation


def extract_segmentation_by_timeframe(sim: Simulation) -> pd.DataFrame | None:
    """Extract segmentation-by-timeframe summary data from a Simulation."""
    if not sim.segmentation_data_by_timeframe:
        return None
    df = (
        pd.concat(sim.segmentation_data_by_timeframe, axis=0, names=["trial"])
        .reorder_levels(["trial", "carrier", "booking_class", "days_prior"])
        .sort_index()
    )
    return df


class SimTabSegByTimeframe(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    segmentation_by_timeframe: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_by_concat_dataframe("segmentation_by_timeframe"),
        extraction_func=extract_segmentation_by_timeframe,
        doc="Segmentation-by-timeframe summary data.",
    )

    @report_figure
    def fig_segmentation_by_timeframe(
        self,
        metric: Literal["bookings", "revenue"],
        *,
        by_carrier: bool | str = True,
        by_class: bool | str = False,
        raw_df: bool = False,
        also_df: bool = False,
        exclude_nogo: bool = True,
    ):
        if self.segmentation_by_timeframe is None:
            raise ValueError("segmentation_by_timeframe not found")
        df = self.segmentation_by_timeframe
        idxs = list(df.index.names)
        if "trial" in idxs:
            idxs.remove("trial")
            df = df.groupby(idxs).mean()
        df = df[metric].stack().rename(metric).reset_index()

        title = f"{metric.title()} by Timeframe"
        if by_class is True:
            title = f"{metric.title()} by Timeframe and Booking Class"
        title_annot = []
        if not by_carrier:
            g = ["days_prior", "segment"]
            if by_class:
                g += ["booking_class"]
            df = df.groupby(g, observed=False)[[metric]].sum().reset_index()
        if by_carrier and not by_class:
            df = df.groupby(["carrier", "days_prior", "segment"], observed=False)[[metric]].sum().reset_index()
        if isinstance(by_carrier, str):
            df = df[df["carrier"] == by_carrier]
            df = df.drop(columns=["carrier"])
            title_annot.append(by_carrier)
            by_carrier = False
        if isinstance(by_class, str):
            df = df[df["booking_class"] == by_class]
            df = df.drop(columns=["booking_class"])
            title_annot.append(f"Class {by_class}")
            by_class = False
        if title_annot:
            title = f"{title} ({', '.join(title_annot)})"
        if exclude_nogo and "carrier" in df.columns:
            df = df[df["carrier"] != "NONE"]
        if raw_df:
            return df

        import altair as alt

        if by_carrier:
            color = "carrier:N"
            color_title = "Carrier"
        elif by_class:
            color = "booking_class:N"
            color_title = "Booking Class"
        else:
            color = "segment:N"
            color_title = "Passenger Type"

        if metric == "revenue":
            metric_fmt = "$,.0f"
        else:
            metric_fmt = ",.2f"

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                color=alt.Color(color).title(color_title),
                x=alt.X("days_prior:O").scale(reverse=True).title("Days Prior to Departure"),
                y=alt.Y(metric),
                tooltip=([alt.Tooltip("carrier").title("Carrier")] if by_carrier else [])
                + ([alt.Tooltip("booking_class").title("Booking Class")] if by_class else [])
                + [
                    alt.Tooltip("segment", title="Passenger Type"),
                    alt.Tooltip("days_prior", title="Days Prior"),
                    alt.Tooltip(metric, format=metric_fmt, title=metric.title()),
                ],
            )
            .properties(
                width=500,
                height=200,
            )
        )
        if by_carrier or by_class:
            chart = chart.facet(
                row=alt.Row("segment:N", title="Passenger Type"),
                title=title,
            )
        else:
            chart = chart.properties(title=title)

        if also_df:
            return chart, df
        return chart

    def fig_bookings_by_timeframe(self, *args, **kwargs):
        return self.fig_segmentation_by_timeframe("bookings", *args, **kwargs)
