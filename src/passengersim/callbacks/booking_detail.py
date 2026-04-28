from __future__ import annotations

import re
import typing
from functools import partial

import altair as alt
import numpy as np
import pandas as pd

from passengersim.callbacks import CallbackData
from passengersim.summaries.generic import GenericSimulationTables

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from passengersim.callbacks import CallbackMixin
    from passengersim.summaries import SimulationTables


def _collect_leg_booking_detail(sim: CallbackMixin, days_prior: int, *, leg_ids: set[int]) -> dict | None:
    """A daily callback function to collect sales for selected legs."""
    if sim.eng.sample < sim.eng.burn_samples:
        # don't collect data during burn period
        return None
    # create a dictionary to store the sales for each leg
    out = {}
    # iterate over all legs in the simulation engine, and collect the current leg sales
    for leg in sim.eng.legs:
        if leg.leg_id not in leg_ids:
            continue
        out[f"leg-{leg.leg_id}-sold"] = leg.sold
    # return the value, which will be end up as "callback_data" in the results
    return out


def collect_leg_booking_detail(leg_ids: set[int]) -> Callable[[CallbackMixin, int], dict | None]:
    leg_ids = set(leg_ids)
    return partial(_collect_leg_booking_detail, leg_ids=leg_ids)


def process_leg_booking_detail(
    data: CallbackData | SimulationTables,
) -> pd.DataFrame:
    if not isinstance(data, CallbackData) and hasattr(data, "callback_data"):
        data = data.callback_data
    df = data.to_dataframe("daily").set_index(["trial", "sample", "days_prior"]).filter(regex="leg-[0-9]+-sold")
    df.columns = [int(re.sub(r"leg-([0-9]+)-sold", r"\1", i)) for i in df.columns]
    df.columns.name = "leg_id"
    return df


def summarize_leg_booking_detail(
    data: CallbackData | GenericSimulationTables,
):
    df = process_leg_booking_detail(data)
    summary = (
        df.groupby(["days_prior"])
        .agg(
            [
                ("mean", "mean"),
                ("q10", lambda x: np.quantile(x, 0.10)),
                ("q25", lambda x: np.quantile(x, 0.25)),
                ("q50", lambda x: np.quantile(x, 0.5)),
                ("q75", lambda x: np.quantile(x, 0.75)),
                ("q90", lambda x: np.quantile(x, 0.90)),
            ]
        )
        .stack(level=0, future_stack=True)
    )
    summary.columns.name = "metric"
    return summary


def fig_leg_booking_detail_rake(
    data: CallbackData | SimulationTables,
    *,
    leg_id: int,
    raw_df: bool = False,
    color: str = "red",
):
    df = summarize_leg_booking_detail(data)
    df = df.query(f"leg_id == {leg_id}").droplevel("leg_id")
    if raw_df:
        return df

    dfc = df.stack(level=0, future_stack=True).rename("value").reset_index()

    median = (
        alt.Chart(dfc.query("metric == 'q50'"))
        .mark_line(color=color, strokeWidth=2.0)
        .encode(
            x=alt.X("days_prior", title="Days Prior to Departure", scale=alt.Scale(reverse=True)),
            y="value",
        )
    )

    q10_q90 = (
        alt.Chart(dfc.query("metric in ['q10', 'q90']"))
        .mark_line(color=color, strokeWidth=1.0, strokeDash=[3, 5])
        .encode(
            x=alt.X("days_prior", title="Days Prior to Departure", scale=alt.Scale(reverse=True)),
            y="value",
            detail="metric",
        )
    )

    q25_q75 = (
        alt.Chart(dfc.query("metric in ['q25', 'q75']"))
        .mark_line(color=color, strokeWidth=1.0, strokeDash=[5, 1])
        .encode(
            x=alt.X("days_prior", title="Days Prior to Departure", scale=alt.Scale(reverse=True)),
            y="value",
            detail="metric",
        )
    )

    return median + q10_q90 + q25_q75


## add accessor and figures to summary class


def _leg_booking_detail(self: GenericSimulationTables):
    if "leg_booking_detail" not in self._data:
        self._data["leg_booking_detail"] = summarize_leg_booking_detail(self)
    return self._data["leg_booking_detail"]


GenericSimulationTables.leg_booking_detail = property(_leg_booking_detail)
GenericSimulationTables.fig_leg_booking_detail_rake = fig_leg_booking_detail_rake
