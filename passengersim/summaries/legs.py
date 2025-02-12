from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from passengersim.reporting import report_figure
from passengersim.utils.nested_dict import from_nested_dict

from .generic import GenericSimulationTables, SimulationTableItem
from .tools import aggregate_by_summing_dataframe, break_on_integer

if TYPE_CHECKING:
    from collections.abc import Collection

    import altair as alt

    from passengersim import Simulation


def extract_legs(sim: Simulation) -> pd.DataFrame | None:
    """Extract leg-level summary data from a Simulation."""
    leg_data = []
    for leg in sim.sim.legs:
        leg_data.append(
            {
                "leg_id": leg.leg_id,
                "carrier": leg.carrier.name,
                "flt_no": leg.flt_no,
                "orig": leg.orig,
                "dest": leg.dest,
                "gt_sold": leg.gt_sold,
                "gt_capacity": leg.gt_capacity,
                "gt_sold_local": leg.gt_sold_local,
                "gt_revenue": leg.gt_revenue,
                "distance": leg.distance,
            }
        )
    if len(leg_data) == 0:
        return None
    return pd.DataFrame(leg_data).set_index("leg_id")


class SimTabLegs(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    legs: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_by_summing_dataframe(
            "legs", ["carrier", "flt_no", "orig", "dest", "distance"]
        ),
        extraction_func=extract_legs,
        computed_fields={
            "avg_load_factor": "100.0 * gt_sold / gt_capacity",
            "avg_local": "100.0 * gt_sold_local / gt_sold",
        },
        doc="Leg-level summary data.",
    )

    @property
    def local_fraction_by_place(self) -> pd.DataFrame:
        """
        The local share of passengers by carrier and place.

        The index of this DataFrame contains all possible places, and the columns
        contain the carriers.

        For each carrier and place, this is the percentage of leg passengers
        on legs arriving or departing from that place that are local passengers
        (i.e. not connecting passengers).  Passengers are considered connecting
        whether the connection is at this place, or at another place.

        If a carrier does not operate any legs to or from a place, or if legs
        are operated but no passengers are booked (which probably indicates a
        config error), the local share is NaN.

        Returns
        -------
        pd.DataFrame
        """
        if "local_fraction_by_place" not in self._data:
            carriers = self.legs.carrier.unique()
            result = defaultdict(dict)
            for carrier in carriers:
                places = set(self.legs.orig.unique()) | set(self.legs.dest.unique())
                for place in places:
                    temp_table = self.legs.loc[
                        (self.legs.carrier == carrier)
                        & ((self.legs.orig == place) | (self.legs.dest == place)),
                        ["gt_sold", "gt_sold_local"],
                    ].sum()
                    if temp_table["gt_sold"] > 0:
                        result[carrier][place] = float(
                            temp_table["gt_sold_local"] / temp_table["gt_sold"]
                        )
                    else:
                        result[carrier][place] = np.nan
            result = from_nested_dict(result, dims=["carrier", "place"]).T
            result = result.sort_index().sort_index(axis=1)
            self._data["local_fraction_by_place"] = result
        return self._data["local_fraction_by_place"]

    def _fig_leg_factor_distribution(
        self,
        title: str,
        leg_attr: str,
        cat_attr: str,
        by_carrier: bool | str = True,
        breakpoints: Collection[int] = None,
        normalize: bool = False,
        *,
        raw_df=False,
    ) -> alt.Chart | pd.DataFrame:
        """
        Figure showing the distribution of leg factors.

        Parameters
        ----------
        title : str
            The title of the figure.
        leg_attr : str
            The attribute of the leg to use for the distribution.  This should be
            a percentage value that ranges 0-100, such as "avg_load_factor" or
            "avg_local".
        cat_attr : str
            The name to use for labeling categories in the resulting figure.
        by_carrier : bool or str, default True
            If True, show the distribution by carrier.  If a string, show the
            distribution for that carrier. If False, show the distribution
            aggregated over all carriers.
        breakpoints : Collection[int, ...], default (25, 30, 35, 40, ..., 90, 95, 100)
            The breakpoints for the load factor ranges, which represent the lowest
            load factor value in each bin. The first and last breakpoints are always
            bounded to 0 and 101, respectively; these bounds can be included explicitly
            or omitted to be included implicitly. Setting the top value to 101 ensures
            that the highest load factor value (100) is included in the last bin.
        normalize : bool, default False
            If True, normalize the frequency by the total number of legs for each
            carrier, so that the sum of the frequencies for each carrier is 1.
        raw_df : bool, default False
            Return the raw data for this figure as a pandas DataFrame, instead
            of generating the figure itself.

        Returns
        -------
        altair.Chart or pd.DataFrame
        """
        if breakpoints is None:
            breakpoints = range(25, 100, 5)  # default breakpoints

        leg_cat = f"{leg_attr}_category"

        new_data = {
            leg_cat: break_on_integer(
                self.legs[leg_attr],
                breakpoints,
                result_name=leg_cat,
            )
        }
        df_for_chart = (
            self.legs.assign(**new_data)
            .groupby(["carrier", leg_cat], observed=False)
            .size()
            .rename("frequency")
            .reset_index()
        )

        if normalize and by_carrier:
            df_for_chart["frequency"] = df_for_chart.groupby("carrier")[
                "frequency"
            ].transform(lambda x: x / x.sum())
        elif not by_carrier:
            df_for_chart = (
                df_for_chart.groupby([leg_cat], observed=False)
                .frequency.sum()
                .reset_index()
            )
            if normalize:
                df_for_chart["frequency"] = (
                    df_for_chart["frequency"] / df_for_chart["frequency"].sum()
                )
        elif isinstance(by_carrier, str):
            df_for_chart = df_for_chart[df_for_chart["carrier"] == by_carrier]
            df_for_chart = df_for_chart.drop(columns=["carrier"])
            if normalize:
                df_for_chart["frequency"] = (
                    df_for_chart["frequency"] / df_for_chart["frequency"].sum()
                )

        freq_label = "Relative Frequency" if normalize else "Count"

        if raw_df:
            return df_for_chart

        import altair as alt

        if by_carrier is True:
            chart = (
                alt.Chart(df_for_chart)
                .mark_bar()
                .encode(
                    x=alt.X(leg_cat, title=cat_attr),
                    y=alt.Y("frequency:Q", title=freq_label),
                    color=alt.Color("carrier:N", title="Carrier"),
                    facet=alt.Facet("carrier:N", columns=2, title="Carrier"),
                    tooltip=[
                        alt.Tooltip("carrier", title="Carrier"),
                        alt.Tooltip(leg_cat, title=cat_attr),
                        alt.Tooltip("frequency", title=freq_label),
                    ],
                )
                .properties(width=300, height=250, title=f"{title} by Carrier")
            )
        else:
            chart = (
                alt.Chart(df_for_chart)
                .mark_bar()
                .encode(
                    x=alt.X(leg_cat, title=cat_attr),
                    y=alt.Y("frequency:Q", title=freq_label),
                    tooltip=[
                        alt.Tooltip("carrier", title="Carrier"),
                        alt.Tooltip(leg_cat, title=cat_attr),
                        alt.Tooltip("frequency", title=freq_label),
                    ],
                )
                .properties(
                    width=600,
                    height=400,
                    title=title if not by_carrier else f"{title} ({by_carrier})",
                )
            )

        return chart

    @report_figure
    def fig_leg_load_factor_distribution(
        self,
        by_carrier: bool | str = True,
        breakpoints: Collection[int] = None,
        normalize: bool = False,
        *,
        raw_df=False,
    ) -> alt.Chart | pd.DataFrame:
        """
        Figure showing the distribution of leg load factors.

        Parameters
        ----------
        by_carrier : bool or str, default True
            If True, show the distribution by carrier.  If a string, show the
            distribution for that carrier. If False, show the distribution
            aggregated over all carriers.
        breakpoints : Collection[int, ...], default (25, 30, 35, 40, ..., 90, 95, 100)
            The breakpoints for the load factor ranges, which represent the lowest
            load factor value in each bin. The first and last breakpoints are always
            bounded to 0 and 101, respectively; these bounds can be included explicitly
            or omitted to be included implicitly. Setting the top value to 101 ensures
            that the highest load factor value (100) is included in the last bin.
        normalize : bool, default False
            If True, normalize the frequency by the total number of legs for each
            carrier, so that the sum of the frequencies for each carrier is 1.
        raw_df : bool, default False
            Return the raw data for this figure as a pandas DataFrame, instead
            of generating the figure itself.

        Returns
        -------
        altair.Chart or pd.DataFrame
        """
        title = "Load Factor Frequency"
        if normalize:
            title = "Load Factor Relative Frequency"
        if isinstance(by_carrier, str):
            title += f" ({by_carrier})"
        return self._fig_leg_factor_distribution(
            title=title,
            leg_attr="avg_load_factor",
            cat_attr="Load Factor Range",
            by_carrier=by_carrier,
            breakpoints=breakpoints,
            normalize=normalize,
            raw_df=raw_df,
        )

    @report_figure
    def fig_leg_local_share_distribution(
        self,
        by_carrier: bool | str = True,
        breakpoints: Collection[int] = None,
        normalize: bool = False,
        *,
        raw_df=False,
    ) -> alt.Chart | pd.DataFrame:
        """
        Figure showing the distribution of leg local shares.

        The local share is the percentage of passengers on a leg that are
        local to the leg's origin and destination (i.e. not connecting).

        Parameters
        ----------
        by_carrier : bool or str, default True
            If True, show the distribution by carrier.  If a string, show the
            distribution for that carrier. If False, show the distribution
            aggregated over all carriers.
        breakpoints : Collection[int, ...], default (0, 10, 20, ..., 90, 100)
            The breakpoints for the load factor ranges, which represent the lowest
            load factor value in each bin. The first and last breakpoints are always
            bounded to 0 and 101, respectively; these bounds can be included explicitly
            or omitted to be included implicitly. Setting the top value to 101 ensures
            that the highest load factor value (100) is included in the last bin.
        normalize : bool, default False
            If True, normalize the frequency by the total number of legs for each
            carrier, so that the sum of the frequencies for each carrier is 1.
        raw_df : bool, default False
            Return the raw data for this figure as a pandas DataFrame, instead
            of generating the figure itself.

        Returns
        -------
        altair.Chart or pd.DataFrame
        """
        if breakpoints is None:
            breakpoints = range(0, 100, 10)
        title = "Local Share Frequency"
        if normalize:
            title = "Local Share Relative Frequency"
        if isinstance(by_carrier, str):
            title += f" ({by_carrier})"
        return self._fig_leg_factor_distribution(
            title=title,
            leg_attr="avg_local",
            cat_attr="Local Share Range",
            by_carrier=by_carrier,
            breakpoints=breakpoints,
            normalize=normalize,
            raw_df=raw_df,
        )

    @report_figure
    def fig_leg_load_v_local(
        self,
        *,
        orig: str | None = None,
        dest: str | None = None,
        place: str | None = None,
        carrier: str | None = None,
        raw_df: bool = False,
        facet_columns: int | None = 2,
        select_leg: bool = False,
    ) -> alt.Chart | pd.DataFrame:
        """
        Figure showing the relationship between leg load factor and local share.

        Parameters
        ----------
        orig : str or None, default None
            Filter the data to only include legs with this origin.
        dest : str or None, default None
            Filter the data to only include legs with this destination.
        place : str or None, default None
            Filter the data to only include legs with this origin or destination.
        carrier : str or None, default None
            Filter the data to only include legs operated by this carrier.
        raw_df : bool, default False
        facet_columns : int or None, default 2
            The number of columns to use for faceting the plot by carrier. If None,
            all facets will appear on one row.
        select_leg : bool, default False
            If True, return an interactive widget that allows the user to select
            specific legs and view their path_legs. This feature is experimental
            and may change without notice.

        Returns
        -------
        altair.Chart or pd.DataFrame
        """
        import altair as alt

        df = self.legs.assign(capacity=self.legs.gt_capacity / self.n_total_samples)
        color = "carrier:N"
        if carrier:
            df = df[df.carrier == carrier]
        if orig:
            df = df[df.orig == orig]
            if len(df.dest.unique()) < 11:
                color = alt.Color("dest:N", title="Destination")
        if dest:
            df = df[df.dest == dest]
            if len(df.orig.unique()) < 11:
                color = alt.Color("orig:N", title="Origin")
        if place:
            df = df[(df.orig == place) | (df.dest == place)]
            df = df.assign(other_place=df.orig.where(df.orig != place, df.dest))
            if len(df.other_place.unique()) < 11:
                color = alt.Color("other_place:N", title="Other Place")

        if raw_df:
            return df

        chart = (
            alt.Chart(df.reset_index())
            .mark_point()
            .encode(
                x=alt.X("avg_local:Q", title="Leg Local Share"),
                y=alt.Y("avg_load_factor:Q", title="Leg Load Factor"),
                size=alt.Size("capacity:Q").scale(
                    zero=True
                ),  # set zero to False for more contrast
                facet=alt.Facet("carrier:N", columns=facet_columns),
                tooltip=[
                    "leg_id",
                    alt.Tooltip("carrier", title="Carrier"),
                    alt.Tooltip("flt_no", title="Flight No"),
                    alt.Tooltip("orig", title="Orig"),
                    alt.Tooltip("dest", title="Dest"),
                    alt.Tooltip("capacity", title="Capacity", format=",.0f"),
                    alt.Tooltip("avg_local", title="Local Share", format=",.2f"),
                    alt.Tooltip("avg_load_factor", title="Load Factor", format=",.2f"),
                ],
                color=color,
            )
        )
        if select_leg:
            point_sel = alt.selection_point(name="point")
            brush_sel = alt.selection_interval(
                name="brush",
                on="[mousedown[event.shiftKey], mouseup] > mousemove",
                translate="[mousedown[event.shiftKey], mouseup] > mousemove!",
            )
            zoom = alt.selection_interval(
                name="zoom",
                bind="scales",
                on="[mousedown[!event.shiftKey], mouseup] > mousemove",
                translate="[mousedown[!event.shiftKey], mouseup] > mousemove!",
            )

            chart_widget = alt.JupyterChart(
                chart.add_params(point_sel).add_params(brush_sel).add_selection(zoom)
            )

            from ipywidgets import VBox

            # table_widget = HTML(value=df.iloc[:0].to_html())
            subchart_widget = alt.JupyterChart(self.fig_select_leg_analysis([]))

            def on_select_point(change):
                sel = change.new.value
                subchart_widget.chart = self.fig_select_leg_analysis(df.index[sel])

            def on_select_brush(change):
                try:
                    sel = change.new.value
                    if sel is None or "avg_local" not in sel:
                        filtered = df.iloc[:0]
                    else:
                        carrier_name = change.new.store[0]["unit"].split("_")[-1]
                        sel_local = sel["avg_local"]
                        sel_load = sel["avg_load_factor"]
                        filter_query = (
                            f"{sel_local[0]} <= `avg_local` <= {sel_local[1]} and "
                            f"{sel_load[0]} <= `avg_load_factor` <= {sel_load[1]}"
                        )
                        filter_query += f" and `carrier` == '{carrier_name}'"
                        filtered = df.query(filter_query)
                    # table_widget.value = filtered.to_html()
                    # table_widget.value = f"<pre>{change.new}</pre>"
                    subchart_widget.chart = self.fig_select_leg_analysis(filtered.index)
                except Exception:
                    # table_widget.value = f"<pre>{e}</pre>"
                    subchart_widget.chart = alt.Chart().mark_point()

            chart_widget.selections.observe(on_select_point, ["point"])
            chart_widget.selections.observe(on_select_brush, ["brush"])
            return VBox(
                [
                    chart_widget,
                    # table_widget,
                    subchart_widget,
                ]
            )

        return chart.interactive()
