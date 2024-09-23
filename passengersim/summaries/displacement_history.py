from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from passengersim.reporting import report_figure

from .generic import GenericSimulationTables, SimulationTableItem
from .tools import combine_sigmas

if TYPE_CHECKING:
    from passengersim import Simulation
    from passengersim.summaries import SimulationTables


def extract_displacement_history(sim: Simulation) -> pd.DataFrame:
    """Extract the average displacement cost history for each carrier."""
    eng = sim.sim
    result = {}
    for carrier in eng.carriers:
        bp = carrier.raw_displacement_cost_trace()
        result[carrier.name] = (
            pd.DataFrame.from_dict(bp, orient="index")
            .sort_index(ascending=False)
            .rename_axis(index="days_prior")
        )
    if result:
        df = pd.concat(result, axis=0, names=["carrier"])
    else:
        df = pd.DataFrame(
            columns=[
                "displacement_mean",
                "displacement_stdev",
            ],
            index=pd.MultiIndex([[], []], [[], []], names=["carrier", "days_prior"]),
        )
    df = df.fillna(0)
    return df


def aggregate_displacement_history(
    summaries: list[SimulationTables],
) -> pd.DataFrame | None:
    frames = []
    for s in summaries:
        frame = getattr(s, "_raw_displacement_history", None)
        if frame is not None:
            frames.append((frame, s.n_total_samples))
    while len(frames) > 1:
        df, df_n = frames[0]
        other, other_n = frames.pop(1)
        df["displacement_stdev"] = np.sqrt(
            combine_sigmas(
                df["displacement_stdev"],
                other["displacement_stdev"],
                df["displacement_mean"],
                other["displacement_mean"],
                df_n,
                other_n,
                ddof=1,
            )
        )
        df["displacement_mean"] = (
            df["displacement_mean"] * df_n + other["displacement_mean"] * other_n
        ) / (df_n + other_n)
        frames[0] = (df, df_n + other_n)
    if frames:
        return frames[0][0]
    return None


class SimTabDisplacementHistory(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of _Generic
    """

    displacement_history: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_displacement_history,
        extraction_func=extract_displacement_history,
        doc="Displacement cost history for each carrier.",
    )

    @report_figure
    def fig_displacement_history(
        self,
        by_carrier: bool | str = True,
        show_stdev: float | bool | None = None,
        raw_df=False,
    ):
        df = self.displacement_history.reset_index()
        color = None
        if isinstance(by_carrier, str):
            df = df[df.carrier == by_carrier]
        elif by_carrier:
            color = "carrier:N"
            if show_stdev is None:
                show_stdev = False
        if show_stdev:
            if show_stdev is True:
                show_stdev = 2
            df["displacement_upper"] = (
                df["displacement_mean"] + show_stdev * df["displacement_stdev"]
            )
            df["displacement_lower"] = (
                df["displacement_mean"] - show_stdev * df["displacement_stdev"]
            ).clip(0, None)
        if raw_df:
            return df

        import altair as alt

        line_encoding = dict(
            x=alt.X("days_prior:Q")
            .scale(reverse=True)
            .title("Days Prior to Departure"),
            y=alt.Y("displacement_mean", title="Displacement Cost"),
        )
        if color:
            line_encoding["color"] = color
        chart = alt.Chart(df)
        fig = chart.mark_line(interpolate="step-before").encode(**line_encoding)
        if show_stdev:
            area_encoding = dict(
                x=alt.X("days_prior:Q")
                .scale(reverse=True)
                .title("Days Prior to Departure"),
                y=alt.Y("displacement_lower:Q", title="Displacement Cost"),
                y2=alt.Y2("displacement_upper:Q", title="Displacement Cost"),
            )
            bound = chart.mark_area(
                opacity=0.1,
                interpolate="step-before",
            ).encode(**area_encoding)
            bound_line = chart.mark_line(
                opacity=0.4, strokeDash=[5, 5], interpolate="step-before"
            ).encode(
                x=alt.X("days_prior:Q")
                .scale(reverse=True)
                .title("Days Prior to Departure")
            )
            top_line = bound_line.encode(
                y=alt.Y("displacement_lower:Q", title="Displacement Cost")
            )
            bottom_line = bound_line.encode(
                y=alt.Y("displacement_upper:Q", title="Displacement Cost")
            )
            fig = fig + bound + top_line + bottom_line
        return fig
