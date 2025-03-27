from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from passengersim.reporting import report_figure

from .generic import GenericSimulationTables, SimulationTableItem
from .tools import aggregate_by_summing_dataframe

if TYPE_CHECKING:
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
    eng = sim.sim
    result = {}
    for carrier in eng.carriers:
        fc = carrier.raw_fare_class_distribution()
        fc_sold = pd.Series(
            {k: v["sold"] for k, v in fc.items()},
            name="frequency",
        )
        fc_rev = pd.Series(
            {k: v["revenue"] for k, v in fc.items()},
            name="frequency",
        )
        result[carrier.name] = pd.concat(
            [fc_sold, fc_rev], axis=1, keys=["sold", "revenue"]
        ).rename_axis(index="booking_class")
    if result:
        df = pd.concat(result, axis=0, names=["carrier"])
    else:
        df = pd.DataFrame(
            columns=["sold", "revenue"],
            index=pd.MultiIndex([[], []], [[], []], names=["carrier", "booking_class"]),
        )
    df = df.fillna(0)
    df["sold"] = df["sold"].astype(int)
    return df


def _fig_fare_class_mix(df: pd.DataFrame, label_threshold: float = 0.06, title=None):
    import altair as alt

    label_threshold_value = (
        df.groupby("carrier", observed=False).avg_sold.sum().max() * label_threshold
    )
    chart = alt.Chart(df, **({"title": title} if title else {})).transform_calculate(
        halfsold="datum.avg_sold / 2.0",
    )
    bars = chart.mark_bar().encode(
        x=alt.X("carrier:N", title="Carrier"),
        y=alt.Y("avg_sold:Q", title="Seats").stack("zero"),
        color="booking_class",
        tooltip=[
            "carrier",
            "booking_class",
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
        order=alt.Order("booking_class:N", sort="descending"),
        tooltip=[
            "carrier",
            "booking_class",
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
    def fig_fare_class_mix(self, raw_df=False, label_threshold=0.06):
        if self.fare_class_mix is not None and self.n_total_samples > 0:
            df = self.fare_class_mix / self.n_total_samples
            df = df.rename(columns={"sold": "avg_sold"})
            df = df.reset_index()[["carrier", "booking_class", "avg_sold"]]
        else:
            return None

        if raw_df:
            return df
        return _fig_fare_class_mix(
            df,
            label_threshold=label_threshold,
            title="Fare Class Mix",
        )
