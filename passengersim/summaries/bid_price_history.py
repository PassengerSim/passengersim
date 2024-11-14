from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from passengersim.reporting import report_figure
from passengersim.utils.nested_dict import from_nested_dict

from .generic import GenericSimulationTables, SimulationTableItem
from .tools import aggregate_by_concat_dataframe

if TYPE_CHECKING:
    from passengersim import Simulation


def extract_bid_price_history(sim: Simulation) -> pd.DataFrame:
    """Compute the average bid price history for each carrier."""
    df = from_nested_dict(
        sim.bid_price_traces, ["trial", "carrier", "days_prior", "measure"]
    ).rename_axis(columns=None)
    df = df.fillna(0)
    return df


# def aggregate_bid_price_history(
#     summaries: list[SimulationTables],
# ) -> pd.DataFrame | None:
#     frames = []
#     for s in summaries:
#         frame = getattr(s, "_raw_bid_price_history", None)
#         if frame is not None:
#             frames.append((frame, s.n_total_samples))
#     while len(frames) > 1:
#         df, df_n = frames[0]
#         other, other_n = frames.pop(1)
#         df["some_cap_bid_price_stdev"] = np.sqrt(
#             combine_sigmas(
#                 df["some_cap_bid_price_stdev"],
#                 other["some_cap_bid_price_stdev"],
#                 df["some_cap_bid_price_mean"],
#                 other["some_cap_bid_price_mean"],
#                 df_n,
#                 other_n,
#                 ddof=1,
#             )
#         )
#         df["bid_price_stdev"] = np.sqrt(
#             combine_sigmas(
#                 df["bid_price_stdev"],
#                 other["bid_price_stdev"],
#                 df["bid_price_mean"],
#                 other["bid_price_mean"],
#                 df_n,
#                 other_n,
#                 ddof=1,
#             )
#         )
#         df["some_cap_bid_price_mean"] = (
#             df["some_cap_bid_price_mean"] * df_n
#             + other["some_cap_bid_price_mean"] * other_n
#         ) / (df_n + other_n)
#         df["bid_price_mean"] = (
#             df["bid_price_mean"] * df_n + other["bid_price_mean"] * other_n
#         ) / (df_n + other_n)
#
#         df["fraction_some_cap"] = (
#             df["fraction_some_cap"] * df_n + other["fraction_some_cap"] * other_n
#         ) / (df_n + other_n)
#         df["fraction_zero_cap"] = (
#             df["fraction_zero_cap"] * df_n + other["fraction_zero_cap"] * other_n
#         ) / (df_n + other_n)
#
#         frames[0] = (df, df_n + other_n)
#     if frames:
#         return frames[0][0]
#     return None


class SimTabBidPriceHistory(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of _Generic
    """

    bid_price_history: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_by_concat_dataframe("bid_price_history"),
        extraction_func=extract_bid_price_history,
        doc="Bid price history for each carrier.",
    )

    @report_figure
    def fig_bid_price_history(
        self,
        by_carrier: bool | str = True,
        show_stdev: float | bool | None = None,
        cap: Literal["some", "zero", None] = None,
        *,
        raw_df=False,
        trial: int | None = None,
        title: str | None = "Bid Price History",
    ):
        if cap is None:
            bp_mean = "bid_price_mean"
        elif cap == "some":
            bp_mean = "some_cap_bid_price_mean"
        elif cap == "zero":
            bp_mean = "zero_cap_bid_price_mean"
        else:
            raise ValueError(f"cap={cap!r} not in ['some', 'zero', None]")
        df = self.bid_price_history.reset_index()
        if trial is not None:
            df = df[df.trial == trial].drop(columns="trial")
        else:
            df = df.groupby(["carrier", "days_prior"]).mean().reset_index()
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
            df["bid_price_upper"] = df[bp_mean] + show_stdev * df["bid_price_stdev"]
            df["bid_price_lower"] = (
                df[bp_mean] - show_stdev * df["bid_price_stdev"]
            ).clip(0, None)
        if raw_df:
            return df

        import altair as alt

        line_encoding = dict(
            x=alt.X("days_prior:Q")
            .scale(reverse=True)
            .title("Days Prior to Departure"),
            y=alt.Y(bp_mean, title="Bid Price"),
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
                y=alt.Y("bid_price_lower:Q", title="Bid Price"),
                y2=alt.Y2("bid_price_upper:Q", title="Bid Price"),
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
                y=alt.Y("bid_price_lower:Q", title="Bid Price")
            )
            bottom_line = bound_line.encode(
                y=alt.Y("bid_price_upper:Q", title="Bid Price")
            )
            fig = fig + bound + top_line + bottom_line
        if title:
            fig = fig.properties(title=title)
        return fig
