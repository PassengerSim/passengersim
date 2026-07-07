"""Compare results across multiple PassengerSim simulations.

The :class:`Contrast` class acts as a named collection of
:class:`~passengersim.summaries.SimulationTables` objects and provides
methods for side-by-side analysis and visualization of simulation outcomes.
"""

import pathlib
import warnings
from collections.abc import Callable
from functools import partial
from typing import Literal, Self

import altair as alt
import numpy as np
import pandas as pd

from passengersim._types import PathLike
from passengersim.reporting import report_figure
from passengersim.summaries import SimulationTables
from passengersim.summaries.demand_to_come import SimTabDemandToCome


class Contrast(dict):
    """A named collection of SimulationTables for side-by-side comparison.

    ``Contrast`` is a :class:`dict` subclass that maps string labels to
    :class:`~passengersim.summaries.SimulationTables` instances. It provides
    methods for aggregating, visualizing, and reporting results across multiple
    PassengerSim simulation runs.

    Module-level functions whose names begin with ``fig_`` are automatically
    accessible as instance methods, with the first ``summaries`` argument
    pre-filled with this object.
    """

    def apply(
        self,
        func: Callable,
        axis: int | Literal["index", "columns", "rows"] = 0,
        warn_if_missing: bool = False,
    ) -> pd.DataFrame | pd.Series:
        """Apply a callable to each non-None entry and concatenate the results.

        Parameters
        ----------
        func : callable
            A function that accepts a :class:`~passengersim.summaries.SimulationTables`
            instance and returns a pandas Series or DataFrame.
        axis : int or {'index', 'columns', 'rows'}, default 0
            The axis along which to concatenate the results. Passed directly to
            :func:`pandas.concat`.
        warn_if_missing : bool, default False
            If True, emit a warning for any key whose value is ``None``.

        Returns
        -------
        pd.DataFrame or pd.Series
            Concatenated results with a ``"source"`` level prepended to the index.
        """
        data = {}
        for k, v in self.items():
            if v is not None:
                data[k] = func(v)
            else:
                if warn_if_missing:
                    warnings.warn(f"no data found for {k!r}", stacklevel=2)
        try:
            return pd.concat(data, axis=axis, names=["source"])
        except TypeError:
            return pd.Series(data).rename_axis(index="source")

    def __getattr__(self, attr):
        """Look up module-level ``fig_*`` functions and bind them to this Contrast.

        Parameters
        ----------
        attr : str
            Attribute name to look up.

        Returns
        -------
        callable
            A partial application of the matching module-level figure function
            with this :class:`Contrast` as the first argument.

        Raises
        ------
        AttributeError
            If ``attr`` is not a module-level ``fig_*`` function.
        """
        if attr.startswith("fig_"):
            g = globals()
            if attr in g:
                return partial(g[attr], self)
                # return lambda *a, **k: g[attr](self, *a, **k)
        raise AttributeError(attr)

    def __dir__(self):
        """Return a sorted list of attributes, including all module-level ``fig_*`` functions.

        Returns
        -------
        list of str
        """
        x = set(super().__dir__())
        x |= {g for g in globals() if g.startswith("fig_")}
        return sorted(x)

    def write_report(self, filename: PathLike, **kwargs) -> pathlib.Path:
        """Write an HTML report summarizing this contrast to a file.

        Parameters
        ----------
        filename : path-like
            Destination path for the HTML report.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`~passengersim.reporting.contrast.to_html`.

        Returns
        -------
        pathlib.Path
            The path of the written report file.
        """
        from passengersim.reporting.contrast import to_html

        return to_html(self, filename, **kwargs)

    def subset(self, keys=None, *, regex: str | None = None) -> Self:
        """Subset the contrast to only include the specified keys.

        Parameters
        ----------
        keys : str or list of str, optional
            The keys to include in the subset. If None, use the `regex` filter.
        regex : str, optional
            A regular expression to filter the keys. If None, use the `keys` filter.

        Returns
        -------
        Contrast
        """
        if keys is not None and regex is not None:
            raise ValueError("cannot use both keys and regex")
        if keys is not None:
            if isinstance(keys, str):
                keys = [keys]
            if not isinstance(keys, list | tuple):
                raise TypeError("keys must be a string or a list of strings")
            result = type(self)({key: self[key] for key in keys if key in self})
        else:
            import re

            matches = []
            for k in self.keys():
                if re.match(regex, k):
                    matches.append(k)
            result = type(self)({k: self[k] for k in matches})
        if len(result) == 0:
            raise ValueError("no matches found")
        return result


class MultiContrast(dict):
    """A collection of :class:`Contrast` objects for multi-scenario comparison.

    ``MultiContrast`` is a :class:`dict` subclass that maps string labels to
    :class:`Contrast` instances. Module-level ``fig_*`` functions are
    automatically accessible as methods; calling them generates one chart per
    :class:`Contrast` and horizontally concatenates the results into a single
    side-by-side Altair figure.
    """

    def __getattr__(self, attr):
        """Look up module-level ``fig_*`` functions and wrap them for multi-contrast display.

        Parameters
        ----------
        attr : str
            Attribute name to look up.

        Returns
        -------
        callable
            A wrapper that calls the figure function on each contained
            :class:`Contrast`, then horizontally concatenates the results via
            :meth:`_hconcat`.

        Raises
        ------
        AttributeError
            If ``attr`` is not a module-level ``fig_*`` function.
        """
        if attr.startswith("fig_"):
            g = globals()
            if attr in g:
                alter_defaults = {}
                if attr in [
                    "fig_carrier_revenues",
                    "fig_carrier_yields",
                    "fig_carrier_total_bookings",
                    "fig_carrier_load_factors",
                ]:
                    alter_defaults["width"] = 300

                def fig_func(*args, **kwargs):
                    figs = {}
                    kwargs.update(alter_defaults)
                    report = kwargs.pop("report", None)
                    trace = kwargs.pop("trace", None)
                    if trace:
                        raise NotImplementedError("trace not implemented")
                    for k, v in self.items():
                        if v is not None:
                            figs[k] = partial(g[attr], v)(*args, **kwargs)
                    fig, title = self._hconcat(figs)
                    if report:
                        report.add_figure(title=title, fig=fig)
                    return fig

                return fig_func
        raise AttributeError(attr)

    def __dir__(self):
        """Return a sorted list of attributes, including all module-level ``fig_*`` functions.

        Returns
        -------
        list of str
        """
        x = set(super().__dir__())
        x |= {g for g in globals() if g.startswith("fig_")}
        return sorted(x)

    @staticmethod
    def _hconcat(charts: dict[str, alt.Chart]) -> tuple[alt.HConcatChart, str]:
        """Horizontally concatenate a dict of Altair charts, prepending keys as sub-titles.

        The global Altair config is stripped from each individual chart and
        re-applied to the concatenated result. Each chart's title is prefixed
        with its key so viewers can identify which :class:`Contrast` produced it.

        Parameters
        ----------
        charts : dict of str to alt.Chart
            Mapping from label (e.g., contrast name) to Altair chart. ``None``
            values trigger a warning and are skipped.

        Returns
        -------
        alt.HConcatChart
            The horizontally concatenated result.
        str
            The base title extracted from the last processed chart (before the
            key prefix was added).

        Raises
        ------
        ValueError
            If ``charts`` is empty or all chart values are ``None``.
        """
        if not charts:
            raise ValueError("no charts to concatenate")
        queue = []
        title = ""
        for k, c in charts.items():
            if c is None:
                warnings.warn(f"no data found for {k!r}", stacklevel=2)
                continue
            config = c._kwds.get("config", alt.Undefined)
            c._kwds["config"] = alt.Undefined
            title = c._kwds.get("title", "")
            c._kwds["title"] = f"{k} {title}"
            queue.append(c)
        if not queue:
            raise ValueError("no charts to concatenate")
        result = alt.hconcat(*queue)
        result._kwds["config"] = config
        return result, title


def _assemble(summaries: dict, base: str, **kwargs) -> pd.DataFrame:
    """Collect and concatenate a named summary table from multiple SimulationTables.

    For each entry in ``summaries`` this function tries (in order):

    1. Call ``fig_{base}(raw_df=True, **kwargs)`` on the summary object.
    2. Access the ``raw_{base}`` attribute of the summary object.
    3. Use the value directly if it is already a :class:`pandas.DataFrame` or
       :class:`pandas.Series`.

    All recovered results are concatenated along axis 0 with a ``"source"``
    level prepended to the index, which is then reset into a column.

    Parameters
    ----------
    summaries : dict of str to SimulationTables
        Mapping from run label to summary tables.
    base : str
        Name suffix identifying the figure or raw-data attribute to retrieve
        (e.g., ``"carrier_revenues"`` resolves to ``fig_carrier_revenues`` or
        ``raw_carrier_revenues``).
    **kwargs
        Additional keyword arguments forwarded to the figure or raw-data
        accessor.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with a ``"source"`` column.

    Raises
    ------
    Exception
        Re-raises the last exception encountered if no data is recovered from
        any of the summaries.
    """
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
    summaries: dict[str, SimulationTables],
    by_carrier: bool | str = True,
    by_class: bool | str = False,
    *,
    by_segment: str | None = None,
    raw_df=False,
    source_labels: bool = False,
    ratio: str | bool = False,
    also_df: bool = False,
) -> alt.Chart | pd.DataFrame:
    """
    Generate a figure contrasting bookings by timeframe for one or more runs.

    Parameters
    ----------
    summaries : dict[str, SimulationTables]
        One or more SimulationTables to compare. The keys of this dictionary are the
        text names used to label the "source" for each set of data in the figure.
    by_carrier : bool or str, default True
        Whether to differentiate carriers by colors (the default) or give the name
        of a particular carrier as a string to filter the results shown in the
        figure to only that one carrier.
    by_class : bool or str, default False
        Whether to differentiate booking class by colors (the default) or give the
        name of a particular booking class as a string to filter the results shown
        in the figure to only that one booking class.
    by_segment : str, optional
        If provided, filter the results to only show this passenger segment.
    raw_df : bool, default False
        Set to true to return the raw dataframe used to generate the figure, instead
        of the figure itself.
    source_labels : bool, default False
        Write source labels above the columns of the figure. Source labels are also
        available as tool tips, but if the figure is being shared as an image without
        tooltips, the source labels may make it easier to interpret.
    ratio : str or bool, default False
        Compute ratios against a reference point and display them in tooltips.
        Pass a source label string to specify the reference, or ``True`` to use
        the first source as the reference.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.

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
    df = _assemble(summaries, "bookings_by_timeframe", by_carrier=by_carrier, by_class=by_class)
    source_order = list(summaries.keys())

    title = "Bookings by Timeframe"
    if by_class is True:
        title = "Bookings by Timeframe and Booking Class"
    title_annot = []
    if isinstance(by_carrier, str):
        title_annot.append(by_carrier)
    if isinstance(by_class, str):
        title_annot.append(f"Class {by_class}")
    if by_segment:
        title_annot.append(by_segment)
    if title_annot:
        title = f"{title} ({', '.join(title_annot)})"

    against = source_order[0]
    ratio_tooltips = ()
    if ratio:
        if isinstance(ratio, str):
            against = ratio
        idx = list({"source", "carrier", "segment", "days_prior", "booking_class"} & set(df.columns))
        df_ = df.set_index(idx)
        ratios = df_.div(df_.query(f"source == '{against}'").droplevel("source")) - 1.0
        ratios.columns = ["ratio"]
        df = df.join(ratios, on=idx)
        ratio_tooltips = (alt.Tooltip("ratio:Q", title=f"vs {against}", format=".3%"),)

    if by_segment:
        df = df[df["segment"] == by_segment]

    if raw_df:
        df.attrs["title"] = title
        return df

    if by_class:
        if isinstance(by_class, str):
            color = alt.Color("source:N", title="Source", sort=source_order).title("Source")
            tooltips = ()
        else:
            color = alt.Color("booking_class:N").title("Booking Class")
            tooltips = (alt.Tooltip("booking_class", title="Booking Class"),)
        chart = alt.Chart(df.sort_values("source", ascending=False))
        chart_1 = chart.mark_bar().encode(
            color=color,
            x=alt.X("days_prior:O").scale(reverse=True).title("Days Prior to Departure"),
            xOffset=alt.XOffset("source:N", title="Source", sort=source_order),
            y=alt.Y("bookings", stack=True),
            tooltip=[
                alt.Tooltip("source:N", title="Source"),
                alt.Tooltip("segment", title="Passenger Segment"),
                *tooltips,
                alt.Tooltip("days_prior", title="DfD"),
                alt.Tooltip("bookings", format=".2f"),
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
            x=alt.X("days_prior:O").scale(reverse=True).title("Days Prior to Departure"),
            xOffset=alt.XOffset("source:N", title="Source", sort=source_order),
            # shape=alt.Shape("source:N", title="Source", sort=source_order),
            y=alt.Y("sum(bookings)", title=None),
        )
        fig = (
            ((chart_1 + chart_2) if source_labels else chart_1)
            .properties(
                width=500,
                height=200,
            )
            .facet(
                row=alt.Row("segment:N", title="Passenger Segment"),
                title=title,
            )
        )
    elif by_carrier is True:
        fig = (
            alt.Chart(df.sort_values("source", ascending=False))
            .mark_bar()
            .encode(
                color=alt.Color("carrier:N").title("Carrier"),
                x=alt.X("days_prior:O").scale(reverse=True).title("Days Prior to Departure"),
                xOffset=alt.XOffset("source:N", title="Source", sort=source_order),
                y=alt.Y("bookings", stack=True),
                tooltip=[
                    alt.Tooltip("source:N", title="Source"),
                    alt.Tooltip("segment", title="Passenger Segment"),
                    alt.Tooltip("carrier", title="Carrier"),
                    alt.Tooltip("days_prior", title="DfD"),
                    alt.Tooltip("bookings", format=".2f"),
                    *ratio_tooltips,
                ],
            )
            .properties(
                width=500,
                height=200,
            )
            .facet(
                row=alt.Row("segment:N", title="Passenger Segment"),
                title=title,
            )
        )
    else:
        fig = (
            alt.Chart(df.sort_values("source", ascending=False), title=title)
            .mark_line()
            .encode(
                color=alt.Color("source:N", title="Source", sort=source_order),
                x=alt.X("days_prior:O").scale(reverse=True).title("Days Prior to Departure"),
                y="bookings",
                strokeDash=alt.StrokeDash("segment").title("Passenger Segment"),
                strokeWidth=alt.StrokeWidth("source:N", title="Source", sort=source_order),
                tooltip=[
                    alt.Tooltip("source:N", title="Source"),
                    alt.Tooltip("segment", title="Passenger Segment"),
                    alt.Tooltip("days_prior", title="DfD"),
                    alt.Tooltip("bookings", format=".2f"),
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
        )
    if also_df:
        return fig, df
    return fig


@report_figure
def fig_segmentation_by_timeframe(
    summaries: dict[str, SimulationTables],
    metric: Literal["bookings", "revenue"],
    by_carrier: bool | str = True,
    by_class: bool | str = False,
    raw_df=False,
    source_labels: bool = False,
    ratio: str | bool = False,
    also_df: bool = False,
) -> alt.Chart | pd.DataFrame:
    """
    Generate a figure contrasting segmentation by timeframe for one or more runs.

    Parameters
    ----------
    summaries : dict[str, SimulationTables]
        One or more SimulationTables to compare. The keys of this dictionary are the
        text names used to label the "source" for each set of data in the figure.
    metric : {'bookings', 'revenue'}
        The metric to display for the segmentation.
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
        Pass a source label string to specify the reference, or ``True`` to use
        the first source as the reference.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.

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
        summaries,
        "segmentation_by_timeframe",
        metric=metric,
        by_carrier=by_carrier,
        by_class=by_class,
    )
    source_order = list(summaries.keys())

    title = f"{metric.title()} by Timeframe"
    if by_class is True:
        title = f"{metric.title()} by Timeframe and Booking Class"
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
        idx = list({"source", "carrier", "segment", "days_prior", "booking_class"} & set(df.columns))
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
            color = alt.Color("source:N", title="Source", sort=source_order).title("Source")
            tooltips = ()
        else:
            color = alt.Color("booking_class:N").title("Booking Class")
            tooltips = (alt.Tooltip("booking_class", title="Booking Class"),)
        chart = alt.Chart(df.sort_values("source", ascending=False))
        chart_1 = chart.mark_bar().encode(
            color=color,
            x=alt.X("days_prior:O").scale(reverse=True).title("Days Prior to Departure"),
            xOffset=alt.XOffset("source:N", title="Source", sort=source_order),
            y=alt.Y(metric, stack=True),
            tooltip=[
                alt.Tooltip("source:N", title="Source"),
                alt.Tooltip("segment", title="Passenger Type"),
                *tooltips,
                alt.Tooltip("days_prior", title="Days Prior"),
                alt.Tooltip(metric, format=".2f"),
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
            x=alt.X("days_prior:O").scale(reverse=True).title("Days Prior to Departure"),
            xOffset=alt.XOffset("source:N", title="Source", sort=source_order),
            # shape=alt.Shape("source:N", title="Source", sort=source_order),
            y=alt.Y(f"sum({metric})", title=None),
        )
        fig = (
            ((chart_1 + chart_2) if source_labels else chart_1)
            .properties(
                width=500,
                height=200,
            )
            .facet(
                row=alt.Row("segment:N", title="Passenger Type"),
                title=title,
            )
        )
    elif by_carrier is True:
        fig = (
            alt.Chart(df.sort_values("source", ascending=False))
            .mark_bar()
            .encode(
                color=alt.Color("carrier:N").title("Carrier"),
                x=alt.X("days_prior:O").scale(reverse=True).title("Days Prior to Departure"),
                xOffset=alt.XOffset("source:N", title="Source", sort=source_order),
                y=alt.Y(metric, stack=True, title=metric.title()),
                tooltip=[
                    alt.Tooltip("source:N", title="Source"),
                    alt.Tooltip("segment", title="Passenger Type"),
                    alt.Tooltip("carrier", title="Carrier"),
                    alt.Tooltip("days_prior", title="Days Prior"),
                    alt.Tooltip(metric, format=".2f"),
                    *ratio_tooltips,
                ],
            )
            .properties(
                width=500,
                height=200,
            )
            .facet(
                row=alt.Row("segment:N", title="Passenger Type"),
                title=title,
            )
        )
    else:
        fig = (
            alt.Chart(df.sort_values("source", ascending=False), title=title)
            .mark_line()
            .encode(
                color=alt.Color("source:N", title="Source", sort=source_order),
                x=alt.X("days_prior:O").scale(reverse=True).title("Days Prior to Departure"),
                y=metric,
                strokeDash=alt.StrokeDash("segment").title("Passenger Type"),
                strokeWidth=alt.StrokeWidth("source:N", title="Source", sort=source_order),
                tooltip=[
                    alt.Tooltip("source:N", title="Source"),
                    alt.Tooltip("segment", title="Passenger Type"),
                    alt.Tooltip("days_prior", title="Days Prior"),
                    alt.Tooltip(metric, format=".2f"),
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
        )
    if also_df:
        return fig, df
    return fig


def _fig_carrier_measure(
    df,
    source_order,
    load_measure: str,
    measure_name: str,
    measure_format: str = ".2f",
    orient: Literal["h", "v"] = "h",
    *,
    title: str | None = None,
    ratio: str | bool = False,
    ratio_all: bool = False,
    ratio_label: bool = True,
    width: int = 500,
):
    """Build a carrier-level bar chart comparing a single metric across sources.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing at least ``source``, ``carrier``, ``rm_system``, and
        the column named by ``load_measure``.
    source_order : list of str
        Ordered list of source labels used to control axis sort order.
    load_measure : str
        Name of the DataFrame column to plot on the value axis.
    measure_name : str
        Human-readable label for the metric, used as the axis title.
    measure_format : str, default ".2f"
        Vega-Lite format string for axis tick labels and text annotations
        (e.g., ``"$.4s"`` for dollar amounts, ``".2f"`` for plain floats).
    orient : {'h', 'v'}, default 'h'
        Chart orientation. ``'h'`` places sources on the y-axis (horizontal
        bars); ``'v'`` places sources on the x-axis (vertical bars).
    title : str, optional
        Chart title. If None, no title is set.
    ratio : str or bool, default False
        If truthy, add a ratio tooltip comparing each bar to a reference source.
        Pass a source-label string to specify the reference explicitly;
        otherwise the first entry in ``source_order`` is used. Ignored when
        ``ratio_all`` is True.
    ratio_all : bool, default False
        If True, compute and show ratio tooltips vs. every source in
        ``source_order``, and add interactive radio buttons to select the
        reference for inline percentage-change labels.
    ratio_label : bool, default True
        If True and ``ratio_all`` is True, annotate bars with an inline
        percentage-change label controlled by the radio-button selection.
    width : int, default 500
        Width of each bar panel in pixels.

    Returns
    -------
    alt.Chart
        An Altair faceted bar chart.
    """
    against = source_order[0]
    if ratio_all:
        queue = []
        for n, a in enumerate(source_order):
            df_ = df.set_index(["source", "carrier"]).drop(columns="rm_system")
            ratios = df_.div(df_.query(f"source == '{a}'").droplevel("source")) - 1.0
            ratios.iloc[:, 0] = ratios.iloc[:, 0].where(ratios.index.get_level_values("source") != a, np.nan)
            ratios.columns = [f"ratio_{n}"]
            queue.append(ratios)
        for q in queue:
            df = df.join(q, on=["source", "carrier"])
    elif ratio:
        if isinstance(ratio, str):
            against = ratio
        df_ = df.set_index(["source", "carrier", "rm_system"])
        ratios = df_.div(df_.query(f"source == '{against}'").droplevel("source")) - 1.0
        ratios.columns = ["ratio_0"]
        df = df.join(ratios, on=["source", "carrier"])

    if ratio_label:
        df["ratio_label"] = df["ratio_0"].apply(lambda x: " " if np.isnan(x) else f"{x:+.1%}")
        for n in range(len(source_order)):
            df[f"ratio_label_{n}"] = df[f"ratio_{n}"].apply(lambda x: " " if np.isnan(x) else f"{x:+.1%}")

        domain_max = df[load_measure].max() * (1 + (0.07 * (500 / width)))

    facet_kwargs = {}
    if title is not None:
        facet_kwargs["title"] = title
    chart = alt.Chart(df)
    tooltips = [
        alt.Tooltip("source", title=None),
        alt.Tooltip("carrier", title="Carrier"),
        alt.Tooltip("rm_system", title="RM System"),
        alt.Tooltip(f"{load_measure}:Q", title=measure_name, format=measure_format),
    ]
    if ratio_all:
        for n, a in enumerate(source_order):
            tooltips.append(
                alt.Tooltip(f"ratio_{n}:Q", title=f"vs {a}", format=".3%"),
            )
    elif ratio:
        tooltips.append(
            alt.Tooltip("ratio_0:Q", title=f"vs {against}", format=".3%"),
        )
    if orient == "v":
        bars = chart.mark_bar().encode(
            color=alt.Color("rm_system:N", title="RM System"),
            x=alt.X("source:N", title=None, sort=source_order),
            y=alt.Y(f"{load_measure}:Q", title=measure_name, axis=alt.Axis(format=measure_format)).stack("zero"),
            tooltip=tooltips,
        )
        text = chart.mark_text(dx=0, dy=3, color="white", baseline="top").encode(
            x=alt.X("source:N", title=None, sort=source_order),
            y=alt.Y(f"{load_measure}:Q", title=measure_name, axis=alt.Axis(format=measure_format)).stack("zero"),
            text=alt.Text(f"{load_measure}:Q", format=measure_format),
        )
        return (
            (bars + text)
            .properties(
                width=55 * len(source_order),
                height=300,
            )
            .facet(column=alt.Column("carrier:N", title="Carrier"), **facet_kwargs)
        )
    else:
        bars = chart.mark_bar().encode(
            color=alt.Color("rm_system:N", title="RM System"),
            y=alt.Y("source:N", title=None, sort=source_order),
            x=alt.X(f"{load_measure}:Q", title=measure_name, axis=alt.Axis(format=measure_format)).stack("zero"),
            tooltip=tooltips,
        )
        text = chart.mark_text(dx=-5, dy=0, color="white", baseline="middle", align="right").encode(
            y=alt.Y("source:N", title=None, sort=source_order),
            x=alt.X(f"{load_measure}:Q", title=measure_name, axis=alt.Axis(format=measure_format)).stack("zero"),
            text=alt.Text(f"{load_measure}:Q", format=measure_format),
        )
        if ratio_label:
            radio_buttons = alt.binding_radio(
                options=[str(n) for n in range(len(source_order))],
                name="Reference Source: ",
                labels=source_order,
            )
            radio_param = alt.param(
                value="0",
                bind=radio_buttons,
                name="ref_source",
            )
            text2 = (
                chart.mark_text(dx=5, dy=0, baseline="middle", align="left")
                .encode(
                    y=alt.Y("source:N", title=None, sort=source_order),
                    x=alt.X(f"{load_measure}:Q", title=measure_name).stack("zero").scale(domain=[0, domain_max]),
                    text=alt.Text("reference_source:N"),
                )
                .transform_calculate(reference_source=f'datum["ratio_label_" + {radio_param.name}]')
                .add_params(radio_param)
            )

        else:
            text2 = None
        return (
            ((bars + text + text2) if ratio_label else (bars + text))
            .properties(
                width=width,
                height=10 + 20 * len(source_order),
            )
            .facet(row=alt.Row("carrier:N", title="Carrier"), **facet_kwargs)
        )


@report_figure
def fig_carrier_revenues(
    summaries,
    raw_df=False,
    orient: Literal["h", "v"] = "h",
    ratio: str | bool = "all",
    *,
    width: int = 500,
    also_df: bool = False,
):
    """
    Generate a figure contrasting carrier revenues for one or more runs.

    Parameters
    ----------
    summaries : dict[str, SimulationTables]
        One or more SimulationTables to compare.
    raw_df : bool, default False
        Return only the raw data used to generate the figure.
    orient : {'h', 'v'}, default 'h'
        Chart orientation; ``'h'`` for horizontal bars, ``'v'`` for vertical.
    ratio : str or bool, default "all"
        Add tooltip(s) giving the percentage change of each carrier's revenue
        relative to a reference source. Pass a source-label string to specify
        the reference, or ``'all'`` to compare against all other sources.
    width : int, default 500
        Width of each bar panel in pixels.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.
    """
    df = _assemble(summaries, "carrier_revenues")
    source_order = list(summaries.keys())
    if raw_df:
        df.attrs["title"] = "Carrier Revenues"
        return df
    fig = _fig_carrier_measure(
        df,
        source_order,
        load_measure="avg_rev",
        measure_name="Revenue ($)",
        measure_format="$.4s",
        orient=orient,
        title="Carrier Revenues",
        ratio=ratio,
        ratio_all=(ratio == "all"),
        width=width,
    )
    if also_df:
        df.attrs["title"] = "Carrier Revenues"
        return fig, df
    return fig


@report_figure
def fig_carrier_yields(
    summaries,
    raw_df=False,
    orient: Literal["h", "v"] = "h",
    ratio: str | bool = "all",
    *,
    width: int = 500,
    also_df: bool = False,
):
    """
    Generate a figure contrasting carrier yields for one or more runs.

    Parameters
    ----------
    summaries : dict[str, SimulationTables]
        One or more SimulationTables to compare.
    raw_df : bool, default False
        Return only the raw data used to generate the figure.
    orient : {'h', 'v'}, default 'h'
        Chart orientation; ``'h'`` for horizontal bars, ``'v'`` for vertical.
    ratio : str or bool, default "all"
        Add tooltip(s) giving the percentage change of each carrier's yield
        relative to a reference source. Pass a source-label string to specify
        the reference, or ``'all'`` to compare against all other sources.
    width : int, default 500
        Width of each bar panel in pixels.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.
    """

    df = _assemble(summaries, "carrier_yields")
    source_order = list(summaries.keys())
    if raw_df:
        df.attrs["title"] = "Carrier Yields"
        return df
    fig = _fig_carrier_measure(
        df,
        source_order,
        load_measure="yield",
        measure_name="Yield ($)",
        measure_format="$.4f",
        orient=orient,
        title="Carrier Yields",
        ratio=ratio,
        ratio_all=(ratio == "all"),
        width=width,
    )
    if also_df:
        df.attrs["title"] = "Carrier Yields"
        return fig, df
    return fig


@report_figure
def fig_carrier_rasm(
    summaries,
    raw_df=False,
    orient: Literal["h", "v"] = "h",
    ratio: str | bool = "all",
    *,
    width: int = 500,
    also_df: bool = False,
):
    """
    Generate a figure contrasting carrier RASM for one or more runs.

    RASM (Revenue per Available Seat Mile) is computed by the underlying
    summary tables and pulled via :func:`_assemble`.

    Parameters
    ----------
    summaries : dict[str, SimulationTables]
        One or more SimulationTables to compare.
    raw_df : bool, default False
        Return only the raw data used to generate the figure.
    orient : {'h', 'v'}, default 'h'
        Chart orientation; ``'h'`` for horizontal bars, ``'v'`` for vertical.
    ratio : str or bool, default "all"
        Add tooltip(s) giving the percentage change of each carrier's RASM
        relative to a reference source. Pass a source-label string to specify
        the reference, or ``'all'`` to compare against all other sources.
    width : int, default 500
        Width of each bar panel in pixels.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.
    """
    df = _assemble(summaries, "carrier_rasm")
    source_order = list(summaries.keys())
    if raw_df:
        df.attrs["title"] = "Carrier RASM"
        return df
    fig = _fig_carrier_measure(
        df,
        source_order,
        load_measure="rasm",
        measure_name="Revenue per Available Seat Mile",
        measure_format="$.4f",
        orient=orient,
        title="Carrier Revenue per Available Seat Mile (RASM)",
        ratio=ratio,
        ratio_all=(ratio == "all"),
        width=width,
    )
    if also_df:
        df.attrs["title"] = "Carrier RASM"
        return fig, df
    return fig


@report_figure
def fig_carrier_local_share(
    summaries,
    load_measure: "Literal['bookings', 'leg_pax']" = "bookings",
    raw_df=False,
    orient: Literal["h", "v"] = "h",
    ratio: str | bool = "all",
    *,
    width: int = 500,
    also_df: bool = False,
):
    """
    Generate a figure contrasting carrier local shares for one or more runs.

    Parameters
    ----------
    summaries : dict[str, SimulationTables]
        One or more SimulationTables to compare.
    load_measure : {'bookings', 'leg_pax'}, default 'bookings'
        Whether to measure local share as a percentage of bookings or of
        leg passengers.
    raw_df : bool, default False
        Return only the raw data used to generate the figure.
    orient : {'h', 'v'}, default 'h'
        Chart orientation; ``'h'`` for horizontal bars, ``'v'`` for vertical.
    ratio : str or bool, default "all"
        Add tooltip(s) giving the percentage change of each carrier's local share
        relative to a reference source. Pass a source-label string to specify
        the reference, or ``'all'`` to compare against all other sources.
    width : int, default 500
        Width of each bar panel in pixels.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.
    """
    measure_name = "Local Percent of Bookings" if load_measure == "bookings" else "Local Percent of Leg Passengers"
    m = "local_pct_bookings" if load_measure == "bookings" else "local_pct_leg_pax"
    df = _assemble(summaries, "carrier_local_share", load_measure=load_measure)
    source_order = list(summaries.keys())
    if raw_df:
        df.attrs["title"] = "Carrier Local Share"
        return df
    fig = _fig_carrier_measure(
        df,
        source_order,
        load_measure=m,
        measure_name=measure_name,
        measure_format=".2f",
        orient=orient,
        title=f"Carrier {measure_name}",
        ratio=ratio,
        ratio_all=(ratio == "all"),
        width=width,
    )
    if also_df:
        df.attrs["title"] = "Carrier Local Share"
        return fig, df
    return fig


@report_figure
def fig_carrier_total_bookings(
    summaries,
    raw_df=False,
    orient: Literal["h", "v"] = "h",
    ratio: str | bool = "all",
    *,
    width: int = 500,
    also_df: bool = False,
):
    """
    Generate a figure contrasting carrier total bookings for one or more runs.

    Parameters
    ----------
    summaries : dict[str, SimulationTables]
        One or more SimulationTables to compare.
    raw_df : bool, default False
        Return only the raw data used to generate the figure.
    orient : {'h', 'v'}, default 'h'
        Chart orientation; ``'h'`` for horizontal bars, ``'v'`` for vertical.
    ratio : str or bool, default "all"
        Add tooltip(s) giving the percentage change of each carrier's bookings
        relative to a reference source. Pass a source-label string to specify
        the reference, or ``'all'`` to compare against all other sources.
    width : int, default 500
        Width of each bar panel in pixels.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.
    """

    df = _assemble(summaries, "carrier_total_bookings")
    # correct for variable name differences; the "avg_sold" column
    # is called "sold" in some older summaries
    if "sold" in df.columns:
        if "avg_sold" in df.columns:
            df["avg_sold"] = df["avg_sold"].fillna(df["sold"])
            df = df.drop(columns="sold")
        else:
            df = df.rename(columns={"sold": "avg_sold"})
    source_order = list(summaries.keys())
    if raw_df:
        df.attrs["title"] = "Carrier Total Bookings"
        return df
    fig = _fig_carrier_measure(
        df,
        source_order,
        load_measure="avg_sold",
        measure_name="Bookings",
        measure_format=".4s",
        orient=orient,
        title="Carrier Total Bookings",
        ratio=ratio,
        ratio_all=(ratio == "all"),
        width=width,
    )
    if also_df:
        df.attrs["title"] = "Carrier Total Bookings"
        return fig, df
    return fig


@report_figure
def fig_carrier_load_factors(
    summaries: dict[str, SimulationTables],
    raw_df: bool = False,
    load_measure: Literal["sys_lf", "avg_leg_lf"] = "sys_lf",
    orient: Literal["h", "v"] = "h",
    ratio: str | bool = "all",
    *,
    width: int = 500,
    also_df: bool = False,
    title: str | None = "_default_",
):
    """
    Generate a figure contrasting carrier load factors for one or more runs.

    Parameters
    ----------
    summaries : dict[str, SimulationTables]
        One or more SimulationTables to compare.
    raw_df : bool, default False
        Return only the raw data used to generate the figure.
    load_measure : {'sys_lf', 'avg_leg_lf'}, default 'sys_lf'
        Which load factor metric to display: system load factor (``'sys_lf'``)
        or average leg load factor (``'avg_leg_lf'``).
    orient : {'h', 'v'}, default 'h'
        Chart orientation; ``'h'`` for horizontal bars, ``'v'`` for vertical.
    ratio : str or bool, default "all"
        Add tooltip(s) giving the percentage change of each carrier's load factor
        relative to a reference source. Pass a source-label string to specify
        the reference, or ``'all'`` to compare against all other sources.
    width : int, default 500
        Width of each bar panel in pixels.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.
    title : str, optional
        Title of the figure. Pass ``None`` for no title; if not provided, a
        default title is derived from ``load_measure``.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.
    """
    measure_name = {
        "sys_lf": "System Load Factor",
        "avg_leg_lf": "Leg Load factor",
    }.get(load_measure, "Load Factor")
    if title == "_default_":
        title = f"Carrier {measure_name}s"
    df = _assemble(summaries, "carrier_load_factors", load_measure=load_measure)
    source_order = list(summaries.keys())
    if raw_df:
        if title:
            df.attrs["title"] = title
        return df
    fig = _fig_carrier_measure(
        df,
        source_order,
        load_measure=load_measure,
        measure_name=measure_name,
        measure_format=".2f",
        orient=orient,
        title=title,
        ratio=ratio,
        ratio_all=(ratio == "all"),
        width=width,
    )
    if also_df:
        if title:
            df.attrs["title"] = title
        return fig, df
    return fig


@report_figure
def fig_fare_class_mix(
    summaries: dict[str, SimulationTables],
    *,
    raw_df: bool = False,
    also_df: bool = False,
    label_threshold: float = 0.06,
) -> alt.Chart | pd.DataFrame | tuple[alt.Chart, pd.DataFrame]:
    """Generate a figure contrasting carrier fare class mix for one or more runs.

    Displays stacked bar charts of average seats sold per booking class for
    each carrier and source, making it easy to compare how fare class mix
    shifts across simulation scenarios.

    Parameters
    ----------
    summaries : dict[str, SimulationTables]
        One or more SimulationTables to compare. The keys are used as source
        labels in the chart.
    raw_df : bool, default False
        If True, return the raw DataFrame instead of the chart.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.
    label_threshold : float, default 0.06
        Minimum fraction of the tallest bar that a segment must represent for
        its inline text label to be shown. Smaller segments are unlabeled to
        avoid clutter.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.
    """
    df = _assemble(summaries, "fare_class_mix")
    source_order = list(summaries.keys())
    if raw_df:
        df.attrs["title"] = "Carrier Fare Class Mix"
        return df
    import altair as alt

    label_threshold_value = df.groupby(["carrier", "source"]).avg_sold.sum().max() * label_threshold
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
        tooltip=[
            "source",
            "booking_class",
            alt.Tooltip("avg_sold", format=".2f"),
        ],
    )
    fig = (
        (bars + text)
        .properties(
            width=200,
            height=300,
        )
        .facet(
            column=alt.Column("carrier:N", title="Carrier"),
            title="Carrier Fare Class Mix",
        )
    )
    if also_df:
        df.attrs["title"] = "Carrier Fare Class Mix"
        return fig, df
    return fig


@report_figure
def fig_od_fare_class_mix(
    summaries: dict[str, SimulationTables],
    orig: str,
    dest: str,
    *,
    raw_df: bool = False,
    also_df: bool = False,
    label_threshold: float = 0.06,
) -> alt.Chart | pd.DataFrame | tuple[alt.Chart, pd.DataFrame]:
    """Generate a figure contrasting O-D fare class mix for one or more runs.

    Like :func:`fig_fare_class_mix` but restricted to a single origin-destination
    market. Displays stacked bar charts of average seats sold per booking class
    for each carrier and source within the specified market.

    Parameters
    ----------
    summaries : dict[str, SimulationTables]
        One or more SimulationTables to compare. The keys are used as source
        labels in the chart.
    orig : str
        Origin airport code for the market of interest.
    dest : str
        Destination airport code for the market of interest.
    raw_df : bool, default False
        If True, return the raw DataFrame instead of the chart.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.
    label_threshold : float, default 0.06
        Minimum fraction of the tallest bar that a segment must represent for
        its inline text label to be shown.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.
    """
    df = _assemble(summaries, "od_fare_class_mix", orig=orig, dest=dest)
    source_order = list(summaries.keys())
    if raw_df:
        df.attrs["title"] = f"Carrier O-D Fare Class Mix ({orig} to {dest})"
        return df
    import altair as alt

    label_threshold_value = df.groupby(["carrier", "source"]).avg_sold.sum().max() * label_threshold
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
        tooltip=[
            "source",
            "booking_class",
            alt.Tooltip("avg_sold", format=".2f"),
        ],
    )
    fig = (
        (bars + text)
        .properties(
            width=200,
            height=300,
        )
        .facet(
            column=alt.Column("carrier:N", title="Carrier"),
            title=f"Carrier O-D Fare Class Mix ({orig} to {dest})",
        )
    )
    if also_df:
        df.attrs["title"] = f"Carrier O-D Fare Class Mix ({orig} to {dest})"
        return fig, df
    return fig


def _fig_forecasts(
    df: pd.DataFrame,
    facet_on: str | None = None,
    y: str = "forecast_mean",
    y_title: str = "Avg Demand Forecast",
    color: str = "booking_class:N",
    rrd_ntype: Literal["O", "Q"] = "O",
) -> alt.Chart:
    """Build a multi-source forecast line chart, optionally faceted.

    Adds a legend-click selection so that clicking a booking class in the
    legend highlights its lines and dims all others.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing at least ``days_prior``, ``source``, ``booking_class``,
        and the column named by ``y``.
    facet_on : str, optional
        If provided, facet the chart on this column (e.g., ``"flt_no"`` or
        ``"path_id"``), with up to 3 columns per row.
    y : str, default "forecast_mean"
        Name of the DataFrame column to plot on the y-axis.
    y_title : str, default "Avg Demand Forecast"
        Y-axis title label.
    color : str, default "booking_class:N"
        Vega-Lite field specification for the color encoding channel.
    rrd_ntype : {'O', 'Q'}, default 'O'
        Vega-Lite data type for the ``days_prior`` field. Use ``'O'`` for
        ordinal (discrete timeframes) and ``'Q'`` for quantitative (daily data).

    Returns
    -------
    alt.Chart
        An Altair line chart (or faceted chart if ``facet_on`` is given) with
        an interactive booking-class selection.
    """
    import altair as alt

    selection = alt.selection_point(name="pick_booking_class", fields=["booking_class"], bind="legend")

    encoding = dict(
        x=alt.X(f"days_prior:{rrd_ntype}").scale(reverse=True).title("Days Prior to Departure"),
        y=alt.Y(f"{y}:Q", title=y_title),
        color="booking_class:N",
        strokeDash=alt.StrokeDash("source:N", title="Source"),
        strokeWidth=alt.StrokeWidth("source:N", title="Source"),
        opacity=alt.condition(selection, alt.value(1.0), alt.value(0.05)),
    )

    if color:
        encoding["color"] = color
    if not facet_on:
        return alt.Chart(df).mark_line().encode(**encoding).add_params(selection)
    else:
        return (
            alt.Chart(df)
            .mark_line()
            .encode(**encoding)
            .facet(
                facet=f"{facet_on}:N",
                columns=3,
            )
            .add_params(selection)
        )


@report_figure
def fig_leg_forecasts(
    summaries: dict[str, SimulationTables],
    raw_df: bool = False,
    by_leg_id: int | None = None,
    by_class: bool | str = True,
    of: Literal["mu", "sigma"] | list[Literal["mu", "sigma"]] = "mu",
    agg_booking_classes: bool = False,
    also_df: bool = False,
) -> alt.Chart | pd.DataFrame | tuple[alt.Chart, pd.DataFrame]:
    """Generate a figure contrasting leg demand forecasts for one or more runs.

    Parameters
    ----------
    summaries : dict[str, SimulationTables]
        One or more SimulationTables to compare.
    raw_df : bool, default False
        If True, return the raw DataFrame instead of the chart.
    by_leg_id : int, optional
        If provided, show forecasts only for this specific leg ID. If None,
        the chart is faceted by flight number.
    by_class : bool or str, default True
        If True, differentiate booking classes by color. If a string, filter
        to only that booking class and color by source instead.
    of : {'mu', 'sigma'} or list thereof, default 'mu'
        Which forecast statistic to display: ``'mu'`` for mean demand forecast,
        ``'sigma'`` for standard deviation. Pass a list to display multiple
        statistics side-by-side.
    agg_booking_classes : bool, default False
        If True, aggregate (sum) across all booking classes before plotting,
        coloring by source rather than booking class.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.
    """
    if isinstance(of, list):
        if raw_df:
            raise NotImplementedError
        fig = fig_leg_forecasts(
            summaries,
            by_leg_id=by_leg_id,
            by_class=by_class,
            of=of[0],
            agg_booking_classes=agg_booking_classes,
        )
        for of_ in of[1:]:
            fig |= fig_leg_forecasts(
                summaries,
                by_leg_id=by_leg_id,
                by_class=by_class,
                of=of_,
                agg_booking_classes=agg_booking_classes,
            )
        title = f"Leg Forecasts {by_leg_id}"
        try:
            if by_leg_id:
                first_summary = next(iter(summaries.values()))
                leg_def = first_summary.legs.loc[by_leg_id]
                title += f": {leg_def['carrier']} {leg_def['flt_no']}"
                title += f" ({leg_def['orig']}-{leg_def['dest']})"
        except Exception:
            raise
        return fig.properties(title=title)
    df = _assemble(summaries, "leg_forecasts", by_leg_id=by_leg_id, by_class=by_class, of=of)
    color = "booking_class:N"
    if isinstance(by_class, str):
        color = "source:N"
    if agg_booking_classes or not by_class:
        color = "source:N"
        if of == "mu":
            df = df.groupby(["source", "leg_id", "days_prior"]).forecast_mean.sum().reset_index()
        elif of == "sigma":

            def sum_sigma(x):
                return np.sqrt(sum(x**2))

            df = df.groupby(["source", "leg_id", "days_prior"]).forecast_stdev.apply(sum_sigma).reset_index()
    if raw_df:
        df.attrs["title"] = "Average Leg Forecasts"
        return df
    fig = _fig_forecasts(
        df,
        facet_on="flt_no" if not isinstance(by_leg_id, int) else None,
        y="forecast_mean" if of == "mu" else "forecast_stdev",
        y_title="Mean Demand Forecast" if of == "mu" else "Std Dev Demand Forecast",
        color=color,
    )
    if also_df:
        df.attrs["title"] = "Average Leg Forecasts"
        return fig, df
    return fig


ForecastOfT = Literal["mu", "sigma", "closed", "adj_price"]


@report_figure
def fig_path_forecasts(
    summaries: dict[str, SimulationTables],
    raw_df: bool = False,
    by_path_id: int | None = None,
    path_names: dict | None = None,
    agg_booking_classes: bool = False,
    by_class: bool | str = True,
    of: ForecastOfT | list[ForecastOfT] = "mu",
    also_df: bool = False,
) -> alt.Chart | pd.DataFrame | tuple[alt.Chart, pd.DataFrame]:
    """Generate a figure contrasting path demand forecasts for one or more runs.

    Parameters
    ----------
    summaries : dict[str, SimulationTables]
        One or more SimulationTables to compare.
    raw_df : bool, default False
        If True, return the raw DataFrame instead of the chart.
    by_path_id : int, optional
        If provided, show forecasts only for this specific path ID. If None,
        the chart is faceted by path ID.
    path_names : dict, optional
        Mapping from integer path IDs to display labels. If provided, path IDs
        in the data are replaced with their corresponding labels before plotting.
    agg_booking_classes : bool, default False
        If True, aggregate (sum or mean, depending on ``of``) across all booking
        classes before plotting, coloring by source rather than booking class.
    by_class : bool or str, default True
        If True, differentiate booking classes by color. If a string, filter
        to only that booking class and color by source instead.
    of : {'mu', 'sigma', 'closed', 'adj_price'} or list thereof, default 'mu'
        Which forecast statistic to display. Pass a list to display multiple
        statistics side-by-side.

        - ``'mu'``: mean demand forecast
        - ``'sigma'``: standard deviation of demand forecast
        - ``'closed'``: mean fraction of timeframe the path was closed
        - ``'adj_price'``: mean adjusted fare
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.
    """
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
        if by_path_id:
            title = f"Path {by_path_id} Forecasts"
        else:
            title = "Path Forecasts"
        try:
            if by_path_id:
                first_summary = next(iter(summaries.values()))
                path_def = first_summary.paths.loc[by_path_id]
                title += f" ({path_def['orig']}~{path_def['dest']})"
                for leg_id in first_summary.path_legs.query(f"path_id == {by_path_id}").leg_id:
                    leg_def = first_summary.legs.loc[leg_id]
                    title += f", {leg_def['carrier']} {leg_def['flt_no']} ({leg_def['orig']}-{leg_def['dest']})"
        except Exception:
            raise
        return fig.properties(title=title)
    df = _assemble(summaries, "path_forecasts", by_path_id=by_path_id, of=of, by_class=by_class)
    list(summaries.keys())
    if path_names is not None:
        df["path_id"] = df["path_id"].apply(lambda x: path_names.get(x, str(x)))
    color = "booking_class:N"
    if isinstance(by_class, str):
        color = "source:N"
    if agg_booking_classes:
        if of == "mu":
            df = df.groupby(["source", "path_id", "days_prior"]).forecast_mean.sum().reset_index()
        elif of == "sigma":

            def sum_sigma(x):
                return np.sqrt(sum(x**2))

            df = df.groupby(["source", "path_id", "days_prior"]).forecast_stdev.apply(sum_sigma).reset_index()
        elif of == "closed":
            df = df.groupby(["source", "path_id", "days_prior"]).forecast_closed_in_tf.mean().reset_index()
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
    elif of == "adj_price":
        y = "adjusted_price"
        y_title = "Mean Adjusted Fare"
    else:
        raise NotImplementedError

    # use ordinal data type for DCP labels unless
    # the underlying data is daily, then use Q
    rrd_ntype: Literal["O", "Q"] = "O"
    if len(df["days_prior"].value_counts()) > 30:
        rrd_ntype = "Q"
    fig = _fig_forecasts(
        df,
        facet_on="path_id" if not isinstance(by_path_id, int) else None,
        y=y,
        y_title=y_title,
        color=color,
        rrd_ntype=rrd_ntype,
    )
    if also_df:
        if of == "mu":
            df.attrs["title"] = "Average Path Forecast Means"
        elif of == "sigma":
            df.attrs["title"] = "Average Path Forecast Standard Deviations"
        elif of == "closed":
            df.attrs["title"] = "Average Path Forecast Closed in Timeframe"
        return fig, df
    return fig


@report_figure
def fig_bid_price_history(
    summaries: dict[str, SimulationTables],
    *,
    by_carrier: bool | str = True,
    show_stdev: float | bool | None = None,
    cap: Literal["some", "zero", None] = None,
    raw_df: bool = False,
    title: str | None = "Bid Price History",
    also_df: bool = False,
    width: int | None = None,
    height: int | None = None,
) -> alt.Chart | pd.DataFrame | tuple[alt.Chart, pd.DataFrame]:
    """Generate a figure contrasting bid price history for one or more runs.

    Plots average bid prices vs. days prior to departure, with one line per
    source. When multiple carriers are present and ``by_carrier`` is not a
    string, the chart is faceted by carrier.

    Parameters
    ----------
    summaries : dict[str, SimulationTables]
        One or more SimulationTables to compare.
    by_carrier : bool or str, default True
        If True, facet the chart by carrier. If a string, filter to only that
        carrier and return a single unfaceted chart.
    show_stdev : float or bool, optional
        If truthy, overlay a shaded band and dashed boundary lines showing the
        bid-price distribution width. Requires ``by_carrier`` to be a specific
        carrier string.
    cap : {'some', 'zero', None}, default None
        Which bid-price capping variant to display. ``None`` uses the raw bid
        price; ``'some'`` uses a partial-cap variant; ``'zero'`` uses the
        zero-capped variant.
    raw_df : bool, default False
        If True, return the raw DataFrame instead of the chart.
    title : str, optional
        Chart title. Defaults to ``"Bid Price History"``. Pass ``None`` for no
        title.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.
    width : int, optional
        Width of each facet panel in pixels. Uses Altair's default if None.
    height : int, optional
        Height of each facet panel in pixels. Uses Altair's default if None.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.

    Raises
    ------
    NotImplementedError
        If ``show_stdev`` is truthy and ``by_carrier`` is not a specific carrier
        string.
    ValueError
        If ``cap`` is not one of ``'some'``, ``'zero'``, or ``None``.
    """
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
            "contrast.fig_bid_price_history with show_stdev requires looking at a single carrier (set `by_carrier`)"
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
        tooltip=[
            alt.Tooltip("days_prior:Q", title="Days Prior to Departure"),
            alt.Tooltip(bp_mean, title="Average Bid Price", format="$.2f"),
            alt.Tooltip("source:N", title="Source"),
        ],
    )
    chart = alt.Chart(df)
    fig = chart.mark_line(interpolate="step-before").encode(**line_encoding)
    if show_stdev:
        area_encoding = dict(
            x=alt.X("days_prior:Q").scale(reverse=True).title("Days Prior to Departure"),
            y=alt.Y("bid_price_lower:Q", title="Bid Price"),
            y2=alt.Y2("bid_price_upper:Q", title="Bid Price"),
            color="source:N",
        )
        bound = chart.mark_area(
            opacity=0.1,
            interpolate="step-before",
        ).encode(**area_encoding)
        bound_line = chart.mark_line(opacity=0.4, strokeDash=[5, 5], interpolate="step-before").encode(
            x=alt.X("days_prior:Q").scale(reverse=True).title("Days Prior to Departure"),
            color="source:N",
        )
        top_line = bound_line.encode(y=alt.Y("bid_price_lower:Q", title="Bid Price"))
        bottom_line = bound_line.encode(y=alt.Y("bid_price_upper:Q", title="Bid Price"))
        fig = fig + bound + top_line + bottom_line
    if not isinstance(by_carrier, str):
        fig = fig.properties(height=height or 150, width=width or 300).facet(facet="carrier:N", columns=2)
        if title:
            fig = fig.properties(title=title)
    else:
        if title:
            title = f"{title} ({by_carrier})"
            fig = fig.properties(title=title)
    if also_df:
        return fig, df
    return fig


@report_figure
def fig_displacement_history(
    summaries: dict[str, SimulationTables],
    *,
    by_carrier: bool | str = True,
    show_stdev: float | bool | None = None,
    raw_df: bool = False,
    title: str | None = "Displacement Cost History",
    also_df: bool = False,
    width: int = 225,
    height: int = 125,
) -> alt.Chart | pd.DataFrame | tuple[alt.Chart, pd.DataFrame]:
    """Generate a figure contrasting displacement cost history for one or more runs.

    Plots average displacement costs vs. days prior to departure, with one line
    per source. When multiple carriers are present and ``by_carrier`` is not a
    string, the chart is faceted by carrier.

    Parameters
    ----------
    summaries : dict[str, SimulationTables]
        One or more SimulationTables to compare.
    by_carrier : bool or str, default True
        If True, facet the chart by carrier. If a string, filter to only that
        carrier and return a single unfaceted chart.
    show_stdev : float or bool, optional
        If truthy, overlay a shaded band and dashed boundary lines showing the
        displacement cost distribution width. Requires ``by_carrier`` to be a
        specific carrier string.
    raw_df : bool, default False
        If True, return the raw DataFrame instead of the chart.
    title : str, optional
        Chart title. Defaults to ``"Displacement Cost History"``. Pass ``None``
        for no title.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.
    width : int, default 225
        Width of each facet panel in pixels.
    height : int, default 125
        Height of each facet panel in pixels.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.

    Raises
    ------
    NotImplementedError
        If ``show_stdev`` is truthy and ``by_carrier`` is not a specific carrier
        string.
    """
    if not isinstance(by_carrier, str) and show_stdev:
        raise NotImplementedError(
            "contrast.fig_displacement_history with show_stdev requires looking at a single carrier (set `by_carrier`)"
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
        tooltip=[
            alt.Tooltip("days_prior:Q", title="Days Prior to Departure"),
            alt.Tooltip("displacement_mean", title="Displacement Cost", format="$.2f"),
            alt.Tooltip("source:N", title="Source"),
        ],
    )
    chart = alt.Chart(df)
    fig = chart.mark_line(interpolate="step-before").encode(**line_encoding)
    if show_stdev:
        area_encoding = dict(
            x=alt.X("days_prior:Q").scale(reverse=True).title("Days Prior to Departure"),
            y=alt.Y("displacement_lower:Q", title="Displacement Cost"),
            y2=alt.Y2("displacement_upper:Q", title="Displacement Cost"),
            color="source:N",
        )
        bound = chart.mark_area(
            opacity=0.1,
            interpolate="step-before",
        ).encode(**area_encoding)
        bound_line = chart.mark_line(opacity=0.4, strokeDash=[5, 5], interpolate="step-before").encode(
            x=alt.X("days_prior:Q").scale(reverse=True).title("Days Prior to Departure"),
            color="source:N",
        )
        top_line = bound_line.encode(y=alt.Y("displacement_lower:Q", title="Displacement Cost"))
        bottom_line = bound_line.encode(y=alt.Y("displacement_upper:Q", title="Displacement Cost"))
        fig = fig + bound + top_line + bottom_line
    if not isinstance(by_carrier, str):
        fig = fig.properties(height=height, width=width).facet(facet="carrier:N", columns=2)
        if title:
            fig = fig.properties(title=title)
    else:
        if title:
            title = f"{title} ({by_carrier})"
            fig = fig.properties(title=title)
    if also_df:
        return fig, df
    return fig


@report_figure
def fig_demand_to_come(
    summaries: Contrast,
    func: Literal["mean", "std"] | list[Literal["mean", "std"]] = "mean",
    *,
    raw_df: bool = False,
    title: str | None = "Demand to Come",
    also_df: bool = False,
) -> alt.Chart | pd.DataFrame | tuple[alt.Chart, pd.DataFrame]:
    """Generate a figure contrasting demand-to-come curves for one or more runs.

    Demand to come is the expected remaining demand (i.e., future bookings)
    at each days-prior-to-departure snapshot, broken out by passenger segment.
    This figure plots those curves across multiple simulation runs for
    easy comparison.

    Parameters
    ----------
    summaries : Contrast
        A :class:`Contrast` containing the simulation results to compare.
    func : {'mean', 'std'} or list thereof, default 'mean'
        Which statistic to display: ``'mean'`` for mean demand to come or
        ``'std'`` for standard deviation. Pass a list to display both
        side-by-side.
    raw_df : bool, default False
        If True, return the raw DataFrame instead of the chart.
    title : str, optional
        Chart title. Defaults to ``"Demand to Come"``. Pass ``None`` for no
        title.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.

    Raises
    ------
    ValueError
        If ``func`` is not ``'mean'``, ``'std'``, or a list containing those
        values.
    """

    def dtc_seg(s):
        """Aggregate demand-to-come over orig/dest dimensions if present.

        If the input index contains ``orig`` or ``dest`` levels they are summed
        away so that only the ``segment`` and ``days_prior`` dimensions remain.
        Returns an empty DataFrame if ``s`` is None.
        """
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

    def get_values(s, which="mean"):
        """Extract mean or std demand-to-come values from a summary object.

        Handles both :class:`~passengersim.summaries.SimulationTables` and
        :class:`~passengersim.summaries.demand_to_come.SimTabDemandToCome`
        inputs, falling back to computing statistics from the raw
        demand-to-come table when a pre-computed summary is not available.
        """
        if isinstance(s, SimulationTables):
            result = getattr(s, "demand_to_come_summary", None)
            if result is None:
                result = dtc_seg(s.demand_to_come).groupby("segment", observed=False)
                result = result.mean() if which == "mean" else result.std()
                result = result.stack()
                return result
            else:
                if which == "mean":
                    return result["mean_future_demand"]
                elif which == "std":
                    return result["stdev_future_demand"]
                else:
                    raise ValueError(f"which must be in [mean, std] not {which}")
        elif isinstance(s, SimTabDemandToCome):
            result = s.demand_to_come_summary
            if which == "mean":
                return result["mean_future_demand"]
            elif which == "std":
                return result["stdev_future_demand"]
            else:
                raise ValueError(f"which must be in [mean, std] not {which}")

    if isinstance(func, list):
        if raw_df:
            raise NotImplementedError
        fig = fig_demand_to_come(summaries, func[0], raw_df=raw_df, title=None)
        for f in func[1:]:
            fig |= fig_demand_to_come(summaries, f, raw_df=raw_df, title=None)
        if title:
            fig = fig.properties(title=title)
        return fig

    if func == "mean":
        y_title = "Mean Demand to Come"
        demand_to_come_by_segment = summaries.apply(lambda s: get_values(s, "mean"), axis=1, warn_if_missing=True)
        demand_to_come_by_segment.index.names = ["segment", "days_prior"]
        df = demand_to_come_by_segment.stack().rename("dtc").reset_index()
    elif func == "std":
        y_title = "Std Dev Demand to Come"
        demand_to_come_by_segment = summaries.apply(lambda s: get_values(s, "std"), axis=1, warn_if_missing=True)
        demand_to_come_by_segment.index.names = ["segment", "days_prior"]
        df = demand_to_come_by_segment.stack().rename("dtc").reset_index()
    else:
        raise ValueError(f"func must be in [mean, std] not {func}")
    if raw_df:
        return df
    fig = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("days_prior:O").scale(reverse=True).title("Days Prior to Departure"),
            y=alt.Y("dtc:Q").title(y_title),
            color="segment:N",
            strokeDash="source:N",
        )
    )
    if title:
        fig = fig.properties(title=title)
    if also_df:
        return fig, df
    return fig


@report_figure
def fig_cp_segmentation(
    summaries: Contrast,
    *,
    raw_df: bool = False,
    title: str | None = "Continuous Price Segmentation",
    also_df: bool = False,
    height: int = 600,
    width: int = 300,
) -> alt.Chart | tuple[alt.Chart, pd.DataFrame] | pd.DataFrame:
    """
    Generate a figure contrasting continuous price segmentation for one or more runs.

    Parameters
    ----------
    summaries : Contrast
        A :class:`Contrast` containing the simulation results to compare. Each
        entry must expose a ``cp_segmentation`` attribute.
    raw_df : bool, default False
        If True, return the raw data used to generate the figure.
    title : str, optional
        The title of the figure. Defaults to ``"Continuous Price Segmentation"``.
        Pass ``None`` for no title.
    also_df : bool, default False
        If True, return a tuple of ``(figure, dataframe)`` instead of just the figure.
    height : int, default 600
        The height of each facet panel of the figure, in pixels.
    width : int, default 300
        The width of each facet panel of the figure, in pixels.

    Returns
    -------
    alt.Chart or pd.DataFrame or tuple[alt.Chart, pd.DataFrame]
        The Altair chart, or the raw DataFrame if ``raw_df`` is True, or a
        ``(chart, dataframe)`` tuple if ``also_df`` is True.
    """
    import altair as alt

    df = pd.concat(
        {k: s.cp_segmentation for k, s in summaries.items() if s.cp_segmentation is not None}, names=["source"]
    )
    source_order = list(summaries.keys())

    df = df.reset_index()[["source", "carrier", "booking_class", "sold", "cp_sold"]]
    if raw_df:
        return df
    df["Zero"] = 0

    base_chart = alt.Chart(df)

    # Add a solid border to the "Continuous Priced" bars using stroke/strokeWidth
    chart = (
        base_chart.mark_bar()
        .encode(
            y=alt.Y("booking_class:N", sort=None, title="Booking Class"),
            x=alt.X("sold:Q", title="Total Sold", stack=False, axis=alt.Axis()),
            color=alt.Color("source:N", title="Source"),
            yOffset=alt.YOffset("source:N", title="Source", sort=source_order),
            tooltip=[
                alt.Tooltip("source:N", title="Source"),
                alt.Tooltip("carrier:N", title="Carrier"),
                alt.Tooltip("booking_class:N", title="Booking Class"),
                alt.Tooltip("sold:Q", title="Total Sold", format=","),
                alt.Tooltip("cp_sold:Q", title="Continuous Priced Sold", format=","),
            ],
        )
        .properties(width=width, height=height)
    )

    chart_cp = (
        base_chart.transform_filter(
            alt.datum.cp_sold > 0  # Only show nonzero counts for cp_sold
        )
        .mark_errorbar(extent="ci", ticks=True)
        .encode(
            y=alt.Y("booking_class:N", sort=source_order, title="Booking Class"),
            x=alt.X("cp_sold:Q", title="Continuous Priced Sold"),
            x2=alt.X2("Zero:Q", title="Zero"),
            color=alt.value("black"),
            yOffset=alt.YOffset("source:N", title="Source", sort=source_order),
            tooltip=[
                alt.Tooltip("source:N", title="Source"),
                alt.Tooltip("carrier:N", title="Carrier"),
                alt.Tooltip("booking_class:N", title="Booking Class"),
                alt.Tooltip("sold:Q", title="Total Sold", format=","),
                alt.Tooltip("cp_sold:Q", title="Continuous Priced Sold", format=","),
            ],
            strokeWidth=alt.value(2),  # Set stroke width for the error bars
        )
        .properties(width=width, height=height)
    )

    fig = (chart + chart_cp).facet(
        facet=alt.Facet("carrier:N", title="Carrier"),
    )

    if title:
        fig = fig.properties(title=title)

    if also_df:
        return fig, df
    else:
        return fig
