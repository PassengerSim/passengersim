import warnings
from collections.abc import Callable
from functools import partial
from typing import Literal

import altair as alt
import numpy as np
import pandas as pd

from .reporting import report_figure
from .summary import SummaryTables


class Contrast(dict):
    def apply(
        self, func: Callable, axis: int | Literal["index", "columns", "rows"] = 0
    ) -> pd.DataFrame | pd.Series:
        data = {}
        for k, v in self.items():
            if v is not None:
                data[k] = func(v)
        try:
            return pd.concat(data, axis=axis, names=["source"])
        except TypeError:
            return pd.Series(data).rename_axis(index="source")

    def __getattr__(self, attr):
        if attr.startswith("fig_"):
            g = globals()
            if attr in g:
                return partial(g[attr], self)
                # return lambda *a, **k: g[attr](self, *a, **k)
        raise AttributeError(attr)

    def __dir__(self):
        x = set(super().__dir__())
        x |= {g for g in globals() if g.startswith("fig_")}
        return sorted(x)


def _assemble(summaries, base, **kwargs):
    summaries_ = {}
    last_exception = RuntimeError("no summaries loaded")
    for k, v in summaries.items():
        if (fun := getattr(v, f"fig_{base}", None)) is not None:
            try:
                summaries_[k] = fun(raw_df=True, **kwargs)
            except Exception as err:
                last_exception = err
                warnings.warn(f"error in getting data from {k!r}: {err}", stacklevel=3)
        elif (raw := getattr(v, f"raw_{base}", None)) is not None:
            try:
                summaries_[k] = raw
            except Exception as err:
                last_exception = err
                warnings.warn(f"error in getting data from {k!r}: {err}", stacklevel=3)
        elif isinstance(v, pd.DataFrame | pd.Series):
            summaries_[k] = v
    if len(summaries_) == 0:
        # no data recovered, re-raise last exception
        raise last_exception
    return pd.concat(summaries_, names=["source"]).reset_index(level="source")


@report_figure
def fig_bookings_by_timeframe(
    summaries: dict[str, SummaryTables],
    by_carrier: bool | str = True,
    by_class: bool | str = False,
    raw_df=False,
    source_labels: bool = False,
    ratio: str | bool = False,
) -> alt.Chart | pd.DataFrame:
    """
    Generate a figure contrasting bookings by timeframe for one or more runs.

    Parameters
    ----------
    summaries : dict[str, SummaryTables]
        One or more SummaryTables to compare. The keys of this dictionary are the
        text names used to label the "source" for each set of data in the figure.
    by_carrier : bool or str, default True
        Whether to differentiate carriers by colors (the default) or give the name
        of a particular carrier as a string to filter the results shown in the
        figure to only that one carrier.
    by_class : bool or str, default False
        Whether to differentiate booking class by colors (the default) or give the
        name of a particular booking class as a string to filter the results shown
        in the figure to only that one booking class.
    raw_df : bool, default False
        Set to true to return the raw dataframe used to generate the figure, instead
        of the figure itself.
    source_labels : bool, default False
        Write source labels above the columns of the figure. Source labels are also
        available as tool tips, but if the figure is being shared as an image without
        tooltips, the source labels may make it easier to interpret.
    ratio : str or bool, default False
        Compute ratios against a reference point and display them in tooltips.

    Other Parameters
    ----------------
    report : xmle.Reporter, optional
        Giving a reporter for this keyword only argument allow you to automatically
        append this figure to the report (in addition to returning it for display
        or other processing).
    trace : pd.ExcelWriter or (pd.ExcelWriter, str), optional
        Write the raw dataframe backing this figure to the Excel workbook.
    """
    if by_carrier is True and by_class is True:
        raise NotImplementedError("comparing by both class and carrier is messy")
    df = _assemble(
        summaries, "bookings_by_timeframe", by_carrier=by_carrier, by_class=by_class
    )
    source_order = list(summaries.keys())

    title = "Bookings by Timeframe"
    if by_class is True:
        title = "Bookings by Timeframe and Booking Class"
    title_annot = []
    if isinstance(by_carrier, str):
        title_annot.append(by_carrier)
    if isinstance(by_class, str):
        title_annot.append(f"Class {by_class}")
    if title_annot:
        title = f"{title} ({', '.join(title_annot)})"

    against = source_order[0]
    ratio_tooltips = ()
    if ratio:
        if isinstance(ratio, str):
            against = ratio
        idx = list(
            {"source", "carrier", "paxtype", "days_prior", "booking_class"}
            & set(df.columns)
        )
        df_ = df.set_index(idx)
        ratios = df_.div(df_.query(f"source == '{against}'").droplevel("source")) - 1.0
        ratios.columns = ["ratio"]
        df = df.join(ratios, on=idx)
        ratio_tooltips = (alt.Tooltip("ratio:Q", title=f"vs {against}", format=".3%"),)

    if raw_df:
        df.attrs["title"] = title
        return df

    if by_class:
        if isinstance(by_class, str):
            color = alt.Color("source:N", title="Source", sort=source_order).title(
                "Source"
            )
            tooltips = ()
        else:
            color = alt.Color("booking_class:N").title("Booking Class")
            tooltips = (alt.Tooltip("booking_class", title="Booking Class"),)
        chart = alt.Chart(df.sort_values("source", ascending=False))
        chart_1 = chart.mark_bar().encode(
            color=color,
            x=alt.X("days_prior:O")
            .scale(reverse=True)
            .title("Days Prior to Departure"),
            xOffset=alt.XOffset("source:N", title="Source", sort=source_order),
            y=alt.Y("sold", stack=True),
            tooltip=[
                alt.Tooltip("source:N", title="Source"),
                alt.Tooltip("paxtype", title="Passenger Type"),
                *tooltips,
                alt.Tooltip("days_prior", title="DfD"),
                alt.Tooltip("sold", format=".2f"),
                *ratio_tooltips,
            ],
        )
        chart_2 = chart.mark_text(
            color="#616161",
            yOffset=-2,
            angle=270,
            fontSize=8,
            baseline="middle",
            align="left",
        ).encode(
            text=alt.Text("source:N", title="Source"),
            x=alt.X("days_prior:O")
            .scale(reverse=True)
            .title("Days Prior to Departure"),
            xOffset=alt.XOffset("source:N", title="Source", sort=source_order),
            # shape=alt.Shape("source:N", title="Source", sort=source_order),
            y=alt.Y("sum(sold)", title=None),
        )
        return (
            ((chart_1 + chart_2) if source_labels else chart_1)
            .properties(
                width=500,
                height=200,
            )
            .facet(
                row=alt.Row("paxtype:N", title="Passenger Type"),
                title=title,
            )
            .configure_title(fontSize=18)
        )
    elif by_carrier is True:
        return (
            alt.Chart(df.sort_values("source", ascending=False))
            .mark_bar()
            .encode(
                color=alt.Color("carrier:N").title("Carrier"),
                x=alt.X("days_prior:O")
                .scale(reverse=True)
                .title("Days Prior to Departure"),
                xOffset=alt.XOffset("source:N", title="Source", sort=source_order),
                y=alt.Y("sold", stack=True),
                tooltip=[
                    alt.Tooltip("source:N", title="Source"),
                    alt.Tooltip("paxtype", title="Passenger Type"),
                    alt.Tooltip("carrier", title="Carrier"),
                    alt.Tooltip("days_prior", title="DfD"),
                    alt.Tooltip("sold", format=".2f"),
                    *ratio_tooltips,
                ],
            )
            .properties(
                width=500,
                height=200,
            )
            .facet(
                row=alt.Row("paxtype:N", title="Passenger Type"),
                title=title,
            )
            .configure_title(fontSize=18)
        )
    else:
        return (
            alt.Chart(df.sort_values("source", ascending=False), title=title)
            .mark_line()
            .encode(
                color=alt.Color("source:N", title="Source", sort=source_order),
                x=alt.X("days_prior:O")
                .scale(reverse=True)
                .title("Days Prior to Departure"),
                y="sold",
                strokeDash=alt.StrokeDash("paxtype").title("Passenger Type"),
                strokeWidth=alt.StrokeWidth(
                    "source:N", title="Source", sort=source_order
                ),
                tooltip=[
                    alt.Tooltip("source:N", title="Source"),
                    alt.Tooltip("paxtype", title="Passenger Type"),
                    alt.Tooltip("days_prior", title="DfD"),
                    alt.Tooltip("sold", format=".2f"),
                    *ratio_tooltips,
                ],
            )
            .properties(
                width=500,
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
            .configure_title(fontSize=18)
        )


def _fig_carrier_measure(
    df,
    source_order,
    load_measure: str,
    measure_name: str,
    measure_format: str = ".2f",
    orient: Literal["h", "v"] = "h",
    title: str | None = None,
    ratio: str | bool = False,
):
    against = source_order[0]
    if ratio:
        if isinstance(ratio, str):
            against = ratio
        df_ = df.set_index(["source", "carrier"])
        ratios = df_.div(df_.query(f"source == '{against}'").droplevel("source")) - 1.0
        ratios.columns = ["ratio"]
        df = df.join(ratios, on=["source", "carrier"])

    facet_kwargs = {}
    if title is not None:
        facet_kwargs["title"] = title
    chart = alt.Chart(df)
    tooltips = [
        alt.Tooltip("source", title=None),
        alt.Tooltip("carrier", title="Carrier"),
        alt.Tooltip(f"{load_measure}:Q", title=measure_name, format=measure_format),
    ]
    if ratio:
        tooltips.append(
            alt.Tooltip("ratio:Q", title=f"vs {against}", format=".3%"),
        )
    if orient == "v":
        bars = chart.mark_bar().encode(
            color=alt.Color("source:N", title="Source"),
            x=alt.X("source:N", title=None, sort=source_order),
            y=alt.Y(f"{load_measure}:Q", title=measure_name).stack("zero"),
            tooltip=tooltips,
        )
        text = chart.mark_text(dx=0, dy=3, color="white", baseline="top").encode(
            x=alt.X("source:N", title=None, sort=source_order),
            y=alt.Y(f"{load_measure}:Q", title=measure_name).stack("zero"),
            text=alt.Text(f"{load_measure}:Q", format=measure_format),
        )
        return (
            (bars + text)
            .properties(
                width=55 * len(source_order),
                height=300,
            )
            .facet(column=alt.Column("carrier:N", title="Carrier"), **facet_kwargs)
            .configure_title(fontSize=18)
        )
    else:
        bars = chart.mark_bar().encode(
            color=alt.Color("source:N", title="Source"),
            y=alt.Y("source:N", title=None, sort=source_order),
            x=alt.X(f"{load_measure}:Q", title=measure_name).stack("zero"),
            tooltip=tooltips,
        )
        text = chart.mark_text(
            dx=-5, dy=0, color="white", baseline="middle", align="right"
        ).encode(
            y=alt.Y("source:N", title=None, sort=source_order),
            x=alt.X(f"{load_measure}:Q", title=measure_name).stack("zero"),
            text=alt.Text(f"{load_measure}:Q", format=measure_format),
        )
        return (
            (bars + text)
            .properties(
                width=500,
                height=10 + 20 * len(source_order),
            )
            .facet(row=alt.Row("carrier:N", title="Carrier"), **facet_kwargs)
            .configure_title(fontSize=18)
        )


@report_figure
def fig_carrier_revenues(
    summaries,
    raw_df=False,
    orient: Literal["h", "v"] = "h",
    ratio: str | bool = True,
):
    """
    Generate a figure contrasting carrier revenues for one or more runs.

    Parameters
    ----------
    summaries : dict[str, SummaryTables]
    raw_df : bool, default False
    orient : {'h', 'v'}, default 'h'
    ratio : bool or str, default True

    Returns
    -------
    alt.Chart or pd.DataFrame
    """
    df = _assemble(summaries, "carrier_revenues")
    source_order = list(summaries.keys())
    if raw_df:
        df.attrs["title"] = "Carrier Revenues"
        return df
    return _fig_carrier_measure(
        df,
        source_order,
        load_measure="avg_rev",
        measure_name="Revenue ($)",
        measure_format="$.4s",
        orient=orient,
        title="Carrier Revenues",
        ratio=ratio,
    )


@report_figure
def fig_carrier_yields(
    summaries,
    raw_df=False,
    orient: Literal["h", "v"] = "h",
    ratio: str | bool = True,
):
    """
    Generate a figure contrasting carrier yields for one or more runs.

    Parameters
    ----------
    summaries : dict[str, SummaryTables]
    raw_df : bool, default False
    orient : {'h', 'v'}, default 'h'
    ratio : bool or str, default True

    Returns
    -------
    alt.Chart or pd.DataFrame
    """

    df = _assemble(summaries, "carrier_yields")
    source_order = list(summaries.keys())
    if raw_df:
        df.attrs["title"] = "Carrier Yields"
        return df
    return _fig_carrier_measure(
        df,
        source_order,
        load_measure="yield",
        measure_name="Yield ($)",
        measure_format="$.4f",
        orient=orient,
        title="Carrier Yields",
        ratio=ratio,
    )


@report_figure
def fig_carrier_total_bookings(
    summaries,
    raw_df=False,
    orient: Literal["h", "v"] = "h",
    ratio: str | bool = True,
):
    """
    Generate a figure contrasting carrier total bookings for one or more runs.

    Parameters
    ----------
    summaries : dict[str, SummaryTables]
    raw_df : bool, default False
    orient : {'h', 'v'}, default 'h'
    ratio : bool or str, default True

    Returns
    -------
    alt.Chart or pd.DataFrame
    """

    df = _assemble(summaries, "carrier_total_bookings")
    source_order = list(summaries.keys())
    if raw_df:
        df.attrs["title"] = "Carrier Total Bookings"
        return df
    return _fig_carrier_measure(
        df,
        source_order,
        load_measure="sold",
        measure_name="Bookings",
        measure_format=".4s",
        orient=orient,
        title="Carrier Total Bookings",
        ratio=ratio,
    )


@report_figure
def fig_carrier_load_factors(
    summaries: dict[str, SummaryTables],
    raw_df: bool = False,
    load_measure: Literal["sys_lf", "avg_leg_lf"] = "sys_lf",
    orient: Literal["h", "v"] = "h",
    ratio: str | bool = True,
):
    """
    Generate a figure contrasting carrier load factors for one or more runs.

    Parameters
    ----------
    summaries : dict[str, SummaryTables]
    raw_df : bool, default False
    load_measure : {'sys_lf', 'avg_leg_lf'}, default 'sys_lf'
    orient : {'h', 'v'}, default 'h'
    ratio : bool or str, default True

    Returns
    -------
    alt.Chart or pd.DataFrame
    """
    measure_name = {
        "sys_lf": "System Load Factor",
        "avg_leg_lf": "Leg Load factor",
    }.get(load_measure, "Load Factor")
    df = _assemble(summaries, "carrier_load_factors", load_measure=load_measure)
    source_order = list(summaries.keys())
    if raw_df:
        df.attrs["title"] = f"Carrier {measure_name}s"
        return df
    return _fig_carrier_measure(
        df,
        source_order,
        load_measure=load_measure,
        measure_name=measure_name,
        measure_format=".2f",
        orient=orient,
        title=f"Carrier {measure_name}s",
        ratio=ratio,
    )


@report_figure
def fig_fare_class_mix(summaries, raw_df=False, label_threshold=0.06):
    df = _assemble(summaries, "fare_class_mix")
    source_order = list(summaries.keys())
    if raw_df:
        df.attrs["title"] = "Carrier Fare Class Mix"
        return df
    import altair as alt

    label_threshold_value = (
        df.groupby(["carrier", "source"]).avg_sold.sum().max() * label_threshold
    )
    chart = alt.Chart(df).transform_calculate(
        halfsold="datum.avg_sold / 2.0",
    )
    bars = chart.mark_bar().encode(
        x=alt.X("source:N", title="Source", sort=source_order),
        y=alt.Y("avg_sold:Q", title="Seats").stack("zero"),
        color="booking_class",
        tooltip=[
            "source",
            "booking_class",
            alt.Tooltip("avg_sold", format=".2f"),
        ],
    )
    text = chart.mark_text(dx=0, dy=3, color="white", baseline="top").encode(
        x=alt.X("source:N", title="Source", sort=source_order),
        y=alt.Y("avg_sold:Q", title="Seats").stack("zero"),
        text=alt.Text("avg_sold:Q", format=".2f"),
        opacity=alt.condition(
            f"datum.avg_sold < {label_threshold_value:.3f}",
            alt.value(0),
            alt.value(1),
        ),
        order=alt.Order("booking_class:N", sort="descending"),
    )
    return (
        (bars + text)
        .properties(
            width=200,
            height=300,
        )
        .facet(
            column=alt.Column("carrier:N", title="Carrier"),
            title="Carrier Fare Class Mix",
        )
        .configure_title(fontSize=18)
    )


def _fig_forecasts(
    df,
    facet_on=None,
    y="forecast_mean",
    y_title="Avg Demand Forecast",
    color="booking_class:N",
    rrd_ntype: Literal["O", "Q"] = "O",
):
    import altair as alt

    encoding = dict(
        x=alt.X(f"days_prior:{rrd_ntype}")
        .scale(reverse=True)
        .title("Days Prior to Departure"),
        y=alt.Y(f"{y}:Q", title=y_title),
        color="booking_class:N",
        strokeDash=alt.StrokeDash("source:N", title="Source"),
        strokeWidth=alt.StrokeWidth("source:N", title="Source"),
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


@report_figure
def fig_leg_forecasts(
    summaries,
    raw_df=False,
    by_flt_no=None,
    by_class: bool | str = True,
    of: Literal["mu", "sigma"] | list[Literal["mu", "sigma"]] = "mu",
    agg_booking_classes: bool = False,
):
    if isinstance(of, list):
        if raw_df:
            raise NotImplementedError
        fig = fig_leg_forecasts(
            summaries,
            by_flt_no=by_flt_no,
            by_class=by_class,
            of=of[0],
            agg_booking_classes=agg_booking_classes,
        )
        for of_ in of[1:]:
            fig |= fig_leg_forecasts(
                summaries,
                by_flt_no=by_flt_no,
                by_class=by_class,
                of=of_,
                agg_booking_classes=agg_booking_classes,
            )
        return fig
    df = _assemble(
        summaries, "leg_forecasts", by_flt_no=by_flt_no, by_class=by_class, of=of
    )
    color = "booking_class:N"
    if isinstance(by_class, str):
        color = "source:N"
    if agg_booking_classes or not by_class:
        color = "source:N"
        if of == "mu":
            df = (
                df.groupby(["source", "flt_no", "days_prior"])
                .forecast_mean.sum()
                .reset_index()
            )
        elif of == "sigma":

            def sum_sigma(x):
                return np.sqrt(sum(x**2))

            df = (
                df.groupby(["source", "flt_no", "days_prior"])
                .forecast_stdev.apply(sum_sigma)
                .reset_index()
            )
    if raw_df:
        df.attrs["title"] = "Average Leg Forecasts"
        return df
    return _fig_forecasts(
        df,
        facet_on="flt_no" if not isinstance(by_flt_no, int) else None,
        y="forecast_mean" if of == "mu" else "forecast_stdev",
        y_title="Mean Demand Forecast" if of == "mu" else "Std Dev Demand Forecast",
        color=color,
    )


@report_figure
def fig_path_forecasts(
    summaries,
    raw_df=False,
    by_path_id=None,
    path_names: dict | None = None,
    agg_booking_classes: bool = False,
    by_class: bool | str = True,
    of: Literal["mu", "sigma", "closed"] = "mu",
):
    if isinstance(of, list):
        if raw_df:
            df = {
                _of: fig_path_forecasts(
                    summaries,
                    by_path_id=by_path_id,
                    path_names=path_names,
                    by_class=by_class,
                    of=of[0],
                    raw_df=raw_df,
                )
                for _of in of
            }
            return pd.concat(df, axis=0, names=["measurement"]).reset_index(0)
        fig = fig_path_forecasts(
            summaries,
            by_path_id=by_path_id,
            path_names=path_names,
            by_class=by_class,
            of=of[0],
        )
        for of_ in of[1:]:
            fig |= fig_path_forecasts(
                summaries,
                by_path_id=by_path_id,
                path_names=path_names,
                by_class=by_class,
                of=of_,
            )
        return fig
    df = _assemble(
        summaries, "path_forecasts", by_path_id=by_path_id, of=of, by_class=by_class
    )
    list(summaries.keys())
    if path_names is not None:
        df["path_id"] = df["path_id"].apply(lambda x: path_names.get(x, str(x)))
    color = "booking_class:N"
    if isinstance(by_class, str):
        color = "source:N"
    if agg_booking_classes:
        if of == "mu":
            df = (
                df.groupby(["source", "path_id", "days_prior"])
                .forecast_mean.sum()
                .reset_index()
            )
        elif of == "sigma":

            def sum_sigma(x):
                return np.sqrt(sum(x**2))

            df = (
                df.groupby(["source", "path_id", "days_prior"])
                .forecast_stdev.apply(sum_sigma)
                .reset_index()
            )
        elif of == "closed":
            df = (
                df.groupby(["source", "path_id", "days_prior"])
                .forecast_closed_in_tf.mean()
                .reset_index()
            )
    if raw_df:
        if of == "mu":
            df.attrs["title"] = "Average Path Forecast Means"
        elif of == "sigma":
            df.attrs["title"] = "Average Path Forecast Standard Deviations"
        elif of == "closed":
            df.attrs["title"] = "Average Path Forecast Closed in Timeframe"
        return df
    if of == "mu":
        y = "forecast_mean"
        y_title = "Mean Demand Forecast"
    elif of == "sigma":
        y = "forecast_stdev"
        y_title = "Std Dev Demand Forecast"
    elif of == "closed":
        y = "forecast_closed_in_tf"
        y_title = "Mean Time Frame Closed of Demand Forecast"
    else:
        raise NotImplementedError

    # use ordinal data type for DCP labels unless
    # the underlying data is daily, then use Q
    rrd_ntype = "O"
    if len(df["days_prior"].value_counts()) > 30:
        rrd_ntype = "Q"
    return _fig_forecasts(
        df,
        facet_on="path_id" if not isinstance(by_path_id, int) else None,
        y=y,
        y_title=y_title,
        color=color,
        rrd_ntype=rrd_ntype,
    )


@report_figure
def fig_bid_price_history(
    summaries,
    by_carrier: bool | str = True,
    show_stdev: float | bool | None = None,
    cap: Literal["some", "zero", None] = None,
    raw_df=False,
):
    if cap is None:
        bp_mean = "bid_price_mean"
    elif cap == "some":
        bp_mean = "some_cap_bid_price_mean"
    elif cap == "zero":
        bp_mean = "zero_cap_bid_price_mean"
    else:
        raise ValueError(f"cap={cap!r} not in ['some', 'zero', None]")

    if not isinstance(by_carrier, str) and show_stdev:
        raise NotImplementedError(
            "contrast.fig_bid_price_history with show_stdev requires "
            "looking at a single carrier (set `by_carrier`)"
        )
    df = _assemble(
        summaries,
        "bid_price_history",
        by_carrier=by_carrier,
        show_stdev=show_stdev,
        cap=cap,
    )
    if raw_df:
        return df

    line_encoding = dict(
        x=alt.X("days_prior:Q").scale(reverse=True).title("Days Prior to Departure"),
        y=alt.Y(bp_mean, title="Bid Price"),
        color="source:N",
    )
    chart = alt.Chart(df)
    fig = chart.mark_line(interpolate="step-before").encode(**line_encoding)
    if show_stdev:
        area_encoding = dict(
            x=alt.X("days_prior:Q")
            .scale(reverse=True)
            .title("Days Prior to Departure"),
            y=alt.Y("bid_price_lower:Q", title="Bid Price"),
            y2=alt.Y2("bid_price_upper:Q", title="Bid Price"),
            color="source:N",
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
            .title("Days Prior to Departure"),
            color="source:N",
        )
        top_line = bound_line.encode(y=alt.Y("bid_price_lower:Q", title="Bid Price"))
        bottom_line = bound_line.encode(y=alt.Y("bid_price_upper:Q", title="Bid Price"))
        fig = fig + bound + top_line + bottom_line
    if not isinstance(by_carrier, str):
        return fig.properties(height=125, width=225).facet(facet="carrier:N", columns=2)
    return fig


@report_figure
def fig_displacement_history(
    summaries,
    by_carrier: bool | str = True,
    show_stdev: float | bool | None = None,
    raw_df=False,
):
    if not isinstance(by_carrier, str) and show_stdev:
        raise NotImplementedError(
            "contrast.fig_displacement_history with show_stdev requires "
            "looking at a single carrier (set `by_carrier`)"
        )
    df = _assemble(
        summaries,
        "displacement_history",
        by_carrier=by_carrier,
        show_stdev=show_stdev,
    )
    if raw_df:
        return df

    line_encoding = dict(
        x=alt.X("days_prior:Q").scale(reverse=True).title("Days Prior to Departure"),
        y=alt.Y("displacement_mean", title="Displacement Cost"),
        color="source:N",
    )
    chart = alt.Chart(df)
    fig = chart.mark_line(interpolate="step-before").encode(**line_encoding)
    if show_stdev:
        area_encoding = dict(
            x=alt.X("days_prior:Q")
            .scale(reverse=True)
            .title("Days Prior to Departure"),
            y=alt.Y("displacement_lower:Q", title="Displacement Cost"),
            y2=alt.Y2("displacement_upper:Q", title="Displacement Cost"),
            color="source:N",
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
            .title("Days Prior to Departure"),
            color="source:N",
        )
        top_line = bound_line.encode(
            y=alt.Y("displacement_lower:Q", title="Displacement Cost")
        )
        bottom_line = bound_line.encode(
            y=alt.Y("displacement_upper:Q", title="Displacement Cost")
        )
        fig = fig + bound + top_line + bottom_line
    if not isinstance(by_carrier, str):
        return fig.properties(height=125, width=225).facet(facet="carrier:N", columns=2)
    return fig


@report_figure
def fig_demand_to_come(
    summaries: Contrast,
    func: Literal["mean", "std"] = "mean",
    raw_df=False,
):
    def dtc_seg(s):
        if s is None:
            return pd.DataFrame(columns=["segment"])
        sum_on = []
        if "orig" in s.index.names:
            sum_on.append("orig")
        if "dest" in s.index.names:
            sum_on.append("dest")
        if sum_on:
            s = s.groupby(s.index.names.difference(sum_on)).sum()
        return s

    if func == "mean":
        y_title = "Mean Demand to Come"
        demand_to_come_by_segment = summaries.apply(
            lambda s: dtc_seg(s.demand_to_come)
            .groupby("segment", observed=False)
            .mean()
            .stack(),
            axis=1,
        )
        df = demand_to_come_by_segment.stack().rename("dtc").reset_index()
    elif func == "std":
        y_title = "Std Dev Demand to Come"
        demand_to_come_by_segment = summaries.apply(
            lambda s: dtc_seg(s.demand_to_come)
            .groupby("segment", observed=False)
            .std()
            .stack(),
            axis=1,
        )
        df = demand_to_come_by_segment.stack().rename("dtc").reset_index()
    else:
        raise ValueError(f"func must be in [mean, std] not {func}")
    if raw_df:
        return df
    return (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("days_prior:O")
            .scale(reverse=True)
            .title("Days Prior to Departure"),
            y=alt.Y("dtc:Q").title(y_title),
            color="segment:N",
            strokeDash="source:N",
        )
    )  # .properties(width=500, height=400)
