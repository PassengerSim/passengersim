from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from passengersim.database import Database
from passengersim.reporting import report_figure

from .generic import DatabaseTableItem, GenericSimulationTables
from .tools import aggregate_by_averaging_dataframe

if TYPE_CHECKING:
    import altair as alt

    from passengersim.contrast import Contrast


def detailed_bookings_by_timeframe(
    cnx: Database,
    *,
    scenario: str | None = None,
    burn_samples: int = 100,
) -> pd.DataFrame:
    """
    Average bookings by carrier, orig, dest, booking class, and timeframe.

    This query requires that the simulation was run while recording supporting
    details (i.e. with the `bookings` or `fare` flags set on `Config.db.write_items`).

    Parameters
    ----------
    cnx : Database
    scenario : str
    burn_samples : int, default 100
        The bookings will be computed ignoring this many samples from the
        beginning of each trial. This argument is nominally ignored by this query
        unless `from_fare_detail` is true, although the simulator will have already
        ignored the burned samples when storing the data in the bookings table.

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe is indexed by `trial`, `carrier`, `orig`, `dest`,
        `booking_class`, and `days_prior`, and has these columns:

        - `avg_sold`: Average number of sales.
        - `avg_business`: Average number of sales to passengers in the business segment.
        - `avg_leisure`: Average number of sales to leisure passengers.
    """
    qry = """
    SELECT trial, carrier, orig, dest, booking_class, days_prior,
           (AVG(sold)) AS avg_sold,
           (AVG(sold_business)) AS avg_business,
           (AVG(sold_leisure)) AS avg_leisure
    FROM (SELECT trial, scenario, carrier, orig, dest, booking_class, days_prior,
                 SUM(sold) AS sold,
                 SUM(sold_business) AS sold_business,
                 SUM(sold - sold_business) AS sold_leisure
          FROM fare_detail LEFT JOIN fare_defs USING (fare_id)
          WHERE
                sample >= ?1
                AND scenario = ?2
          GROUP BY trial, sample, carrier, orig, dest, booking_class, days_prior) a
    GROUP BY carrier, orig, dest, booking_class, days_prior, trial
    ORDER BY carrier, orig, dest, booking_class, days_prior, trial;
    """
    if scenario is None:
        qry = qry.replace("AND scenario = ?2", "")
        params = (burn_samples,)
    else:
        params = (burn_samples, scenario)

    return cnx.dataframe(qry, params).set_index(["trial", "carrier", "orig", "dest", "booking_class", "days_prior"])


class SimTabSegmentationDetail(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of _Generic
    """

    segmentation_detail: pd.DataFrame = DatabaseTableItem(
        aggregation_func=aggregate_by_averaging_dataframe("segmentation_detail"),
        query_func=detailed_bookings_by_timeframe,
        doc="Segmentation detail.",
    )

    @report_figure
    def fig_segmentation_detail(
        self,
        *,
        by_carrier: bool | str = True,
        by_class: bool | str = False,
        orig: str | None = None,
        dest: str | None = None,
        raw_df: bool = False,
        also_df: bool = False,
    ) -> alt.Chart | pd.DataFrame | tuple[alt.Chart, pd.DataFrame]:
        """
        Plot the segmentation detail data.

        Parameters
        ----------
        by_carrier : bool or str, default True
            If True, group by carrier. If a string, filter by carrier.
        by_class : bool or str, default False
            If True, group by booking class. If a string, filter by booking class.
        orig : str, optional
            Filter by origin.
        dest : str, optional
            Filter by destination.
        raw_df : bool, default False
            If True, return the raw dataframe instead of the figure.
        also_df : bool, default False
            If True, return the dataframe as well as the figure.

        Returns
        -------
        alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
            The segmentation detail figure or dataframe.
        """
        if self.segmentation_detail is None:
            raise ValueError("segmentation_detail not found")
        metric = "bookings"
        df = self.segmentation_detail

        # convert to sales by timeframe
        df = df.groupby(df.index.names[:-1])[["avg_business", "avg_leisure"]].diff(-1)
        df = df.groupby(df.index.names[:-1])[["avg_business", "avg_leisure"]].shift()
        df = df.query("days_prior != 0")

        # rename columns
        df = df.rename(columns={"avg_business": "business", "avg_leisure": "leisure"})
        df.columns.name = "segment"
        df = df.stack().rename("bookings").to_frame()

        if orig is not None:
            df = df.query("orig == @orig")
            df.index = df.index.droplevel("orig")
        else:
            gb = [i for i in df.index.names if i != "orig"]
            df = df.groupby(gb).sum()
        if dest is not None:
            df = df.query("dest == @dest")
            df.index = df.index.droplevel("dest")
        else:
            gb = [i for i in df.index.names if i != "dest"]
            df = df.groupby(gb).sum()

        idxs = list(df.index.names)
        if "trial" in idxs:
            idxs.remove("trial")
            df = df.groupby(idxs).mean()
        df = df.reset_index()

        title = "Detailed Segmentation by Timeframe"
        if orig and dest:
            title = f"{title} ({orig}~{dest})"
        elif orig and not dest:
            title = f"{title} (Orig={orig})"
        elif not orig and dest:
            title = f"{title} (Dest={dest})"
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


# Figure for Contrast
def fig_segmentation_detail(
    summaries: Contrast,
    *,
    by_carrier: bool | str = True,
    by_class: bool | str = False,
    orig: str | None = None,
    dest: str | None = None,
    raw_df: bool = False,
    also_df: bool = False,
    width: int | None = 400,
    height: int | None = 180,
) -> alt.Chart | pd.DataFrame | tuple[alt.Chart, pd.DataFrame]:
    """
    Plot the segmentation detail data.

    Parameters
    ----------
    summaries : Contrast
        The contrast object containing the segmentation detail data.
    by_carrier : bool or str, default True
        If True, group by carrier. If a string, filter by carrier.
    by_class : bool or str, default False
        If True, group by booking class. If a string, filter by booking class.
    orig : str, optional
        Filter by origin.
    dest : str, optional
        Filter by destination.
    raw_df : bool, default False
        If True, return the raw dataframe instead of the figure.
    also_df : bool, default False
        If True, return the dataframe as well as the figure.
    width : int, optional
        The width of the figure. Default is 400.
    height : int, optional
        The height of the figure. Default is 180.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The segmentation detail figure or dataframe.
    """
    dfs = {
        k: v.fig_segmentation_detail(
            by_carrier=by_carrier,
            by_class=by_class,
            orig=orig,
            dest=dest,
            raw_df=True,
        )
        for k, v in summaries.items()
    }
    df = pd.concat(dfs, names=["source"]).reset_index()
    if raw_df:
        return df

    metric = "bookings"
    if metric == "revenue":
        metric_fmt = "$,.0f"
    else:
        metric_fmt = ",.2f"

    title = "Detailed Segmentation by Timeframe"
    title_annot = []
    if orig and dest:
        title_annot.append(f"{orig}~{dest}")
    elif orig and not dest:
        title_annot.append(f"Orig={orig}")
    elif not orig and dest:
        title_annot.append(f"Dest={dest}")

    if isinstance(by_carrier, str):
        # this filter is already applied in the query against each source
        # and can now be removed
        title_annot.append(by_carrier)
        by_carrier = False

    title = f"{title} ({', '.join(title_annot)})"

    if by_carrier:
        color = "carrier:N"
        color_title = "Carrier"
    elif by_class:
        color = "booking_class:N"
        color_title = "Booking Class"
    else:
        color = "segment:N"
        color_title = "Passenger Type"

    import altair as alt

    fig = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            xOffset=alt.XOffset("source:N"),
            x=alt.X("days_prior:O").scale(reverse=True).title("Days Prior to Departure"),
            y=alt.Y("bookings:Q").title("Bookings"),
            color=alt.Color(color).title(color_title),
            tooltip=[
                alt.Tooltip("source", title="Source"),
            ]
            + ([alt.Tooltip("carrier").title("Carrier")] if by_carrier else [])
            + ([alt.Tooltip("booking_class").title("Booking Class")] if by_class else [])
            + [
                alt.Tooltip("segment", title="Passenger Type"),
                alt.Tooltip("days_prior", title="Days Prior"),
                alt.Tooltip(metric, format=metric_fmt, title=metric.title()),
            ],
        )
    )
    if by_carrier or by_class:
        fig = fig.facet(
            row=alt.Row("segment:N", title="Passenger Type"),
            title=title,
        )
        if width:
            fig.spec.width = width
        if height:
            fig.spec.height = height
    else:
        fig = fig.properties(title=title, width=width, height=height)

    if also_df:
        return fig, df
    return fig
