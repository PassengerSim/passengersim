from __future__ import annotations

import re
import typing
from functools import partial

import altair as alt
import numpy as np
import pandas as pd

from passengersim.callbacks import CallbackData
from passengersim.summaries.generic import GenericSimulationTables
from passengersim.utils.colors import DarkOrange, DarkPurple

if typing.TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from passengersim.callbacks import CallbackMixin
    from passengersim.summaries import SimulationTables


def _collect_leg_bid_price_detail(
    sim: CallbackMixin, days_prior: int, *, leg_ids: set[int], include_sold_out: bool = True
) -> dict | None:
    """A daily callback function to collect bid price for selected legs."""
    if sim.eng.sample < sim.eng.burn_samples:
        # don't collect data during burn period
        return None
    # create a dictionary to store the bid prices for each leg
    out = {}
    # iterate over all legs in the simulation engine, and collect the current leg bid prices
    for leg in sim.eng.legs:
        if leg.leg_id not in leg_ids:
            continue
        if include_sold_out:
            out[f"leg-{leg.leg_id}-bidprice"] = leg.report_bid_price
        else:
            if leg.bid_price > 9.9e9:
                continue  # leg is sold out, exclude
            out[f"leg-{leg.leg_id}-bidprice"] = leg.bid_price
    # return the value, which will be end up as "callback_data" in the results
    return out


def collect_leg_bid_price_detail(leg_ids: set[int]) -> Callable[[CallbackMixin, int], dict | None]:
    leg_ids = set(leg_ids)
    return partial(_collect_leg_bid_price_detail, leg_ids=leg_ids)


def process_leg_bid_price_detail(
    data: CallbackData | SimulationTables,
) -> pd.DataFrame:
    if not isinstance(data, CallbackData) and hasattr(data, "callback_data"):
        data = data.callback_data
    df = data.to_dataframe("daily").set_index(["trial", "sample", "days_prior"]).filter(regex="leg-[0-9]+-bidprice")
    df.columns = [int(re.sub(r"leg-([0-9]+)-bidprice", r"\1", i)) for i in df.columns]
    df.columns.name = "leg_id"
    return df


def summarize_leg_bid_price_detail(
    data: CallbackData | SimulationTables,
):
    df = process_leg_bid_price_detail(data)

    _quantile_probs = [0.10, 0.25, 0.5, 0.75, 0.90]
    _metric_names = ["mean", "q10", "q25", "q50", "q75", "q90"]

    def _compute_stats(group):
        arr = group.values
        m = np.mean(arr, axis=0, keepdims=True)
        q = np.quantile(arr, _quantile_probs, axis=0)
        return pd.DataFrame(np.vstack([m, q]), index=_metric_names, columns=group.columns)

    summary = df.groupby("days_prior").apply(_compute_stats)
    summary.columns.name = "leg_id"
    result = summary.stack(future_stack=True).unstack(level=1)
    result.columns.name = "metric"
    return result


def fig_leg_bid_price_detail_rake(
    data: CallbackData | SimulationTables,
    *,
    leg_id: int,
    raw_df: bool = False,
    color: str = DarkPurple,
    mean_color: str | None = DarkOrange,
):
    df = summarize_leg_bid_price_detail(data)
    df = df.query(f"leg_id == {leg_id}").droplevel("leg_id")
    if raw_df:
        return df

    dfc = df.stack(level=0, future_stack=True).rename("value").reset_index()

    # if mean_color:
    #     coloring = dict(
    #         color=alt.Color(
    #             "metric",
    #             scale=alt.Scale(
    #                 range=[color, mean_color],
    #                 domain=["'Median'", "'Mean'"],
    #             ),
    #         )
    #     )
    # else:
    #     coloring = {}

    median = (
        alt.Chart(dfc.query("metric == 'q50'"))
        .transform_calculate(
            legend_label="'Median'"  # Creates a static label for the legend
        )
        .mark_line(strokeWidth=2.0, color=color)
        .encode(
            x=alt.X("days_prior", title="Days Prior to Departure", scale=alt.Scale(reverse=True)),
            y=alt.Y("value", title="Bid Price", axis=alt.Axis(format="$,.0f")),
            tooltip=[
                alt.Tooltip("days_prior", title="Days Prior to Departure"),
                alt.Tooltip("value", title="Median Bid Price", format="$,.2f"),
            ],
        )
    )

    q10_q90 = (
        alt.Chart(dfc.query("metric in ['q10', 'q90']"))
        .transform_calculate(
            legend_label="'Outer Deciles'"  # Creates a static label for the legend
        )
        .mark_line(strokeWidth=1.0, strokeDash=[3, 1], color=color)
        .encode(
            x=alt.X("days_prior", title="Days Prior to Departure", scale=alt.Scale(reverse=True)),
            y=alt.Y("value", title="Bid Price", axis=alt.Axis(format="$,.0f")),
            detail="metric",
            tooltip=[
                alt.Tooltip("days_prior", title="Days Prior to Departure"),
                alt.Tooltip("value", title="Outer Decile Bid Price", format="$,.2f"),
            ],
        )
    )

    q25_q75 = (
        alt.Chart(dfc.query("metric in ['q25', 'q75']"))
        .transform_calculate(
            legend_label="'Outer Quartiles'"  # Creates a static label for the legend
        )
        .mark_line(strokeWidth=1.0, color=color, strokeDash=[5, 1])
        .encode(
            x=alt.X("days_prior", title="Days Prior to Departure", scale=alt.Scale(reverse=True)),
            y=alt.Y("value", title="Bid Price", axis=alt.Axis(format="$,.0f")),
            detail="metric",
            tooltip=[
                alt.Tooltip("days_prior", title="Days Prior to Departure"),
                alt.Tooltip("value", title="Outer Quartile Bid Price", format="$,.2f"),
            ],
        )
    )

    if not mean_color:
        return median + q10_q90 + q25_q75

    mean = (
        alt.Chart(dfc.query("metric == 'mean'"))
        .transform_calculate(
            legend_label="'Mean'"  # Creates a static label for the legend
        )
        .mark_line(strokeWidth=3.0, strokeDash=[1, 1, 4, 1, 1, 4], color=mean_color)
        .encode(
            x=alt.X("days_prior", title="Days Prior to Departure", scale=alt.Scale(reverse=True)),
            y=alt.Y("value", title="Bid Price", axis=alt.Axis(format="$,.0f")),
            tooltip=[
                alt.Tooltip("days_prior", title="Days Prior to Departure"),
                alt.Tooltip("value", title="Mean Bid Price", format="$,.2f"),
            ],
            # **coloring,
        )
    )
    return median + q10_q90 + q25_q75 + mean


def fig_leg_bid_price_history(
    self: SimulationTables,
    carrier: str,
    *,
    measure: Literal["mean", "q10", "q25", "q50", "q75", "q90", "median"],
    haul_category_labels: tuple[str, ...] | None = ("a. Short: ", "b. Medium: ", "c. Long: ", "d. Longest: "),
    opacity: float = 0.25,
    max_rows: int = 5000,
) -> alt.Chart:

    if measure == "median":
        measure = "q50"
    measure_labels = {
        "mean": "Mean",
        "q10": "10th Percentile",
        "q25": "25th Percentile",
        "q50": "Median",
        "q75": "75th Percentile",
        "q90": "90th Percentile",
    }
    measure_label = measure_labels.get(measure, measure)
    title = f"{measure_label} Leg Bid Prices by Days Prior to Departure ({carrier})"

    defs = self.leg_defs.query(f"carrier == {carrier!r}").copy()
    defs["leg_label"] = defs["carrier"] + defs["fltno"].astype(str) + ":" + defs["orig"] + "-" + defs["dest"]
    if haul_category_labels is not None:
        defs["haul"] = pd.qcut(self.leg_defs["distance"], q=len(haul_category_labels))
        df = self.leg_bid_price_detail.join(defs[["haul", "leg_label"]], on="leg_id", how="inner")
        df["haul"] = df["haul"].cat.rename_categories(
            pd.Index(haul_category_labels) + df["haul"].cat.categories.astype(str)
        )
        facet = dict(facet=alt.Facet("haul", columns=2))
        title = f"{measure_label} Leg Bid Prices by Days Prior to Departure and Length of Haul ({carrier})"
    else:
        df = self.leg_bid_price_detail.join(defs[["leg_label"]], on="leg_id", how="inner")
        facet = dict()

    if len(df) > max_rows:
        df = df.query("days_prior in @self.config.dcps")
    if len(df) > max_rows:
        df = df.sample(max_rows)
    chart = (
        alt.Chart(df.reset_index())
        .mark_line(opacity=opacity)
        .encode(
            x=alt.X("days_prior", scale=alt.Scale(reverse=True), title="Days Prior to Departure"),
            y=alt.Y(measure, title=f"{measure_label} Bid Price", axis=alt.Axis(format="$,.0f")),
            detail="leg_id",
            tooltip=[
                alt.Tooltip("leg_id", title="Leg ID"),
                alt.Tooltip("leg_label", title=" "),
                alt.Tooltip("days_prior", title="Days Prior"),
                alt.Tooltip(measure, title=f"{measure_label} Bid Price", format="$,.2f"),
            ],
            **facet,
        )
    )
    return chart.properties(title=title)


## add accessor and figures to summary class


def _leg_bid_price_detail(self: GenericSimulationTables):
    if "leg_booking_detail" not in self._data:
        self._data["leg_booking_detail"] = summarize_leg_bid_price_detail(self)
    return self._data["leg_booking_detail"]


GenericSimulationTables.leg_bid_price_detail = property(_leg_bid_price_detail)
GenericSimulationTables.fig_leg_bid_price_detail_rake = fig_leg_bid_price_detail_rake
GenericSimulationTables.fig_leg_bid_price_history = fig_leg_bid_price_history
