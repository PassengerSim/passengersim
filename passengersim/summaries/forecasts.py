from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from passengersim.database import common_queries
from passengersim.reporting import report_figure

from .generic import DatabaseTableItem, GenericSimulationTables
from .tools import aggregate_by_averaging_dataframe, aggregate_by_concat_dataframe

if TYPE_CHECKING:
    import altair as alt


def _fig_forecasts(
    df,
    facet_on=None,
    y="forecast_mean",
    color="booking_class:N",
    y_title="Avg Demand Forecast",
) -> alt.TopLevelMixin:
    import altair as alt

    encoding = dict(
        x=alt.X("days_prior:O").scale(reverse=True).title("Days Prior to Departure"),
        y=alt.Y(f"{y}:Q", title=y_title),
    )
    if color:
        encoding["color"] = color
    if not facet_on:
        return alt.Chart(df).mark_line().encode(**encoding)
    else:
        return (
            alt.Chart(df)
            .mark_line()
            .encode(**encoding)
            .facet(
                facet=f"{facet_on}:N",
                columns=3,
            )
        )


class SimTabForecasts(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of _Generic
    """

    path_forecasts: pd.DataFrame = DatabaseTableItem(
        aggregation_func=aggregate_by_averaging_dataframe("path_forecasts"),
        query_func=common_queries.path_forecasts,
        doc="Path forecasts.",
    )

    leg_forecasts: pd.DataFrame = DatabaseTableItem(
        aggregation_func=aggregate_by_averaging_dataframe("leg_forecasts"),
        query_func=common_queries.leg_forecasts,
        doc="Leg forecasts.",
    )

    edgar: pd.DataFrame = DatabaseTableItem(
        aggregation_func=aggregate_by_concat_dataframe("edgar"),
        query_func=common_queries.edgar,
        doc="EDGAR forecast accuracy measurement.",
    )

    @report_figure
    def fig_path_forecasts(
        self,
        by_path_id: bool | int = True,
        *,
        by_class: bool | str = True,
        of: Literal["mu", "sigma", "closed", "adj_price"] = "mu",
        raw_df: bool = False,
    ):
        if isinstance(of, list):
            if raw_df:
                df = {
                    _of: self.fig_path_forecasts(
                        by_path_id=by_path_id,
                        by_class=by_class,
                        of=_of,
                        raw_df=raw_df,
                    )
                    for _of in of
                }
                return pd.concat(df, axis=0, names=["measurement"]).reset_index(0)
            fig = self.fig_path_forecasts(
                by_path_id=by_path_id,
                by_class=by_class,
                of=of[0],
            )
            for of_ in of[1:]:
                fig |= self.fig_path_forecasts(
                    by_path_id=by_path_id,
                    by_class=by_class,
                    of=of_,
                )
            if by_path_id:
                title = f"Path {by_path_id} Forecasts"
            else:
                title = "Path Forecasts"
            try:
                if by_path_id:
                    path_def = self.paths.loc[by_path_id]
                    title += f" ({path_def['orig']}~{path_def['dest']})"
                    for leg_id in self.path_legs.query(f"path_id == {by_path_id}").leg_id:
                        leg_def = self.legs.loc[leg_id]
                        title += f", {leg_def['carrier']} {leg_def['flt_no']} " f"({leg_def['orig']}-{leg_def['dest']})"
            except Exception:
                raise
            return fig.properties(title=title)

        if self.path_forecasts is None:
            raise ValueError("the path_forecasts table is not available")
        of_columns = {
            "mu": "forecast_mean",
            "sigma": "forecast_stdev",
            "closed": "forecast_closed_in_tf",
            "adj_price": "adjusted_price",
        }
        y = of_columns.get(of)
        y_titles = {
            "mu": "Mean Demand Forecast",
            "sigma": "Std Dev Demand Forecast",
            "closed": "Closed in Timeframe",
            "adj_price": "Adjusted Price",
        }
        y_title = y_titles.get(of)
        columns = [
            "path_id",
            "booking_class",
            "days_prior",
            y,
        ]
        df = self.path_forecasts.reset_index()[columns]
        df = df.query("days_prior > 0")
        color = "booking_class:N"
        if isinstance(by_path_id, int) and by_path_id is not True:
            df = df[df.path_id == by_path_id]
        if isinstance(by_class, str):
            df = df[df.booking_class == by_class]
            color = None
        if raw_df:
            return df
        facet_on = None
        if by_path_id is True:
            facet_on = "path_id"
        return _fig_forecasts(df, facet_on=facet_on, y=y, color=color, y_title=y_title)

    @report_figure
    def fig_leg_forecasts(
        self,
        by_leg_id: bool | int = True,
        *,
        by_class: bool | str = True,
        of: Literal["mu", "sigma", "closed"] | list[Literal["mu", "sigma", "closed"]] = "mu",
        raw_df=False,
    ):
        if isinstance(of, list):
            if raw_df:
                df0 = self.fig_leg_forecasts(
                    by_leg_id=by_leg_id,
                    by_class=by_class,
                    of=of[0],
                    raw_df=True,
                )
                df0 = df0.set_index(list(df0.columns[:-1]))
                for of_ in of[1:]:
                    df1 = self.fig_leg_forecasts(
                        by_leg_id=by_leg_id,
                        by_class=by_class,
                        of=of_,
                        raw_df=True,
                    )
                    df1 = df1.set_index(list(df1.columns[:-1]))
                    df0 = pd.concat([df0, df1], axis=1)
                return df0.reset_index()
            fig = self.fig_leg_forecasts(
                by_leg_id=by_leg_id,
                by_class=by_class,
                of=of[0],
            )
            for of_ in of[1:]:
                fig |= self.fig_leg_forecasts(
                    by_leg_id=by_leg_id,
                    by_class=by_class,
                    of=of_,
                )
            return fig
        if of == "mu":
            y = "forecast_mean"
            y_title = "Mean Demand Forecast"
        elif of == "sigma":
            y = "forecast_stdev"
            y_title = "Std Dev Demand Forecast"
        elif of == "closed":
            y = "forecast_closed_in_tf"
            y_title = "Closed in Time Frame"
        else:
            raise ValueError(f"Unknown 'of' value: {of}")
        columns = [
            "carrier",
            "leg_id",
            "booking_class",
            "days_prior",
            y,
        ]
        if self.leg_forecasts is None:
            raise ValueError("the leg_forecasts table is not available")
        df = self.leg_forecasts.reset_index()[columns]
        df = df.query("days_prior > 0")
        color = "booking_class:N"
        if isinstance(by_leg_id, int) and by_leg_id is not True:
            df = df[df.leg_id == by_leg_id]
        if isinstance(by_class, str):
            df = df[df.booking_class == by_class]
            color = None
        if raw_df:
            return df
        return _fig_forecasts(
            df,
            facet_on=None,
            y=y,
            color=color,
            y_title=y_title,
        )
