from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from passengersim.reporting import report_figure

from .generic import GenericSimulationTables, SimulationTableItem
from .tools import aggregate_by_summing_dataframe

if TYPE_CHECKING:
    import altair as alt

    from passengersim import Simulation


def extract_fare_class_mix(sim: Simulation) -> pd.DataFrame:
    """
    Compute a fare class distribution report.

    This report is a dataframe, with index values giving the carrier and
    booking class, and columns giving the number sold (bookings) and revenue
    observed during the simulation (excluding any burn period). The sales are
    counts of passengers not legs, so a passenger on a connecting itinerary
    only counts once.
    """
    eng = sim.eng
    result2 = []
    for carrier in eng.carriers:
        fc = carrier.raw_fare_class_distribution()
        for k, v in fc.items():
            result2.append(
                {
                    "carrier": carrier.name,
                    "fare_class": k,
                    "booking_class": v["booking_class"],
                    "brand": v["brand"],
                    "sold": v["sold"],
                    "revenue": v["revenue"],
                }
            )

    if len(result2) > 0:
        df2 = pd.DataFrame(result2).set_index(["carrier", "fare_class", "booking_class", "brand"])
    else:
        df2 = pd.DataFrame(
            columns=["sold", "revenue"],
            index=pd.MultiIndex([[], []], [[], []], names=["carrier", "fare_class", "booking_class", "brand"]),
        )

    df2 = df2.fillna(0)
    df2["sold"] = df2["sold"].astype(int)
    return df2


def _fig_fare_class_mix(df: pd.DataFrame, label_threshold: float = 0.06, title=None, group_by=""):
    import altair as alt

    if group_by in ["", "fare_class"]:
        group_key = "fare_class"
    elif group_by == "booking_class":
        group_key = "booking_class"
        df = df.groupby(["carrier", "booking_class"], as_index=True).agg(avg_sales=("sold", "sum"))
        title += " - by Booking Class"
    elif group_by == "brand":
        group_key = "brand"
        df = df.groupby(["carrier", "brand"], as_index=True).agg(avg_sales=("sold", "sum"))
        title += " - by Brand"
    else:
        raise ValueError("Invalid value for group_by")

    label_threshold_value = df.groupby("carrier", observed=False).avg_sold.sum().max() * label_threshold
    chart = alt.Chart(df, **({"title": title} if title else {})).transform_calculate(
        halfsold="datum.avg_sold / 2.0",
    )
    bars = chart.mark_bar().encode(
        x=alt.X("carrier:N", title="Carrier"),
        y=alt.Y("avg_sold:Q", title="Seats").stack("zero"),
        color=group_key,
        tooltip=[
            "carrier",
            group_key,
            alt.Tooltip("avg_sold", format=".2f"),
        ],
    )
    text = chart.mark_text(dx=0, dy=3, color="white", baseline="top").encode(
        x=alt.X("carrier:N", title="Carrier"),
        y=alt.Y("avg_sold:Q", title="Seats").stack("zero"),
        text=alt.Text("avg_sold:Q", format=".2f"),
        opacity=alt.condition(
            f"datum.avg_sold < {label_threshold_value:.3f}",
            alt.value(0),
            alt.value(1),
        ),
        order=alt.Order("fare_class:N", sort="descending"),
        tooltip=[
            "carrier",
            group_key,
            alt.Tooltip("avg_sold", format=".2f"),
        ],
    )
    return (
        (bars + text)
        .properties(
            width=400,
            height=300,
        )
        .configure_axis(
            labelFontSize=12,
            titleFontSize=12,
        )
        .configure_legend(
            titleFontSize=12,
            labelFontSize=15,
        )
    )


class SimTabFareClassMix(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    fare_class_mix: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_by_summing_dataframe("fare_class_mix"),
        extraction_func=extract_fare_class_mix,
        doc="Fare class mix data.",
    )

    @report_figure
    def fig_fare_class_mix(
        self,
        *,
        raw_df: bool = False,
        also_df: bool = False,
        label_threshold: float = 0.06,
        title: str | None = "Fare Class Mix",
        group_by: str | None = "",
    ) -> alt.Chart | pd.DataFrame | tuple[alt.Chart, pd.DataFrame]:
        """
        Plot the fare class mix data.

        Parameters
        ----------
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
        if self.fare_class_mix is not None and self.n_total_samples > 0:
            df = self.fare_class_mix / self.n_total_samples
            df = df.rename(columns={"sold": "avg_sold"})
            df = df.reset_index()[["carrier", "fare_class", "booking_class", "brand", "avg_sold"]]
        else:
            return None

        if raw_df:
            return df
        fig = _fig_fare_class_mix(df, label_threshold=label_threshold, title=title, group_by=group_by)
        if also_df:
            return fig, df
        return fig
