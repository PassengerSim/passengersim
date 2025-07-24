from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import pandas as pd
import yaml

from passengersim.config import Config
from passengersim.utils.bootstrap import BootstrapHtml
from passengersim.utils.filenaming import filename_with_timestamp

from .report import Elem

if TYPE_CHECKING:
    from passengersim.summaries import SimulationTables


def to_html(
    summary: SimulationTables,
    filename: str | None = None,
    *,
    cfg: Config | None = None,
    make_dirs: bool = True,
    extra: tuple | None = None,
) -> pathlib.Path:
    """
    Write a summary to an HTML file.

    Parameters
    ----------
    summary : SimulationTables
    filename : Path-like, optional
        If not provided, the filename will be taken from the run config.  If
        it is also not defined there, a ValueError will be raised.  A timestamp
        will be appended to the filename, so that each report is unique and
        does not overwrite previous reports.
    cfg : Config, optional
        If not provided, the configuration will be taken from the summary
    make_dirs : bool, optional
        If True, create any necessary directories.
    extra : tuple, optional
        Additional data to include in the report.  Each item in the tuple should
        either a section or subsection title, or a tuple of (title, func), or
        just a function.  If a function is provided, it should take the summary
        as its only argument and return a figure (altair.Chart or xmle.Elem) or
        table (pandas.DataFrame).

    Returns
    -------
    pathlib.Path
        The path to the written file.  This includes any appended timestamp.
    """
    if cfg is None:
        cfg = summary.config
        if cfg is None:
            # no config provided, so use a default one
            cfg = Config()
            cfg.outputs.html.filename = None
            cfg.outputs.html.configs = []

    if filename is None:
        if cfg.outputs.html.filename is None:
            raise ValueError("No filename provided")
        filename = pathlib.Path(cfg.outputs.html.filename)

    rpt = BootstrapHtml(title=cfg.outputs.html.title or cfg.scenario)

    rpt.new_section("Results")

    if cfg.outputs.html.carrier_revenues:
        rpt.add_figure(summary.fig_carrier_revenues(also_df=True))

    if cfg.outputs.html.carrier_total_bookings:
        rpt.add_figure(summary.fig_carrier_total_bookings(also_df=True))

    if cfg.outputs.html.carrier_load_factors:
        rpt.add_figure(summary.fig_carrier_load_factors(also_df=True))

    if cfg.outputs.html.carrier_yields:
        rpt.add_figure(summary.fig_carrier_yields(also_df=True))

    if cfg.outputs.html.carrier_rasm:
        rpt.add_figure(summary.fig_carrier_rasm(also_df=True))

    if cfg.outputs.html.carrier_table:
        rpt.add_table("Carrier Data", summary.carriers.T)

    if cfg.outputs.html.fare_class_mix:
        rpt.add_figure(summary.fig_fare_class_mix(also_df=True))

    if cfg.outputs.html.bookings_by_timeframe:
        rpt.add_figure(summary.fig_bookings_by_timeframe(also_df=True))
        if cfg.outputs.html.bookings_by_timeframe is True:
            carriers = list(cfg.carriers.keys())
        else:
            carriers = cfg.outputs.html.bookings_by_timeframe
        for c in carriers:
            rpt.add_figure(summary.fig_segmentation_by_timeframe("bookings", by_carrier=c, also_df=True))
            rpt.add_figure(summary.fig_segmentation_by_timeframe("bookings", by_carrier=c, by_class=True, also_df=True))

    if cfg.outputs.html.segmentation_by_timeframe_table:
        rpt.add_table(
            "Segmentation by Timeframe Data",
            Elem.from_string(summary.segmentation_by_timeframe.to_html(max_rows=10_000)),
            collapsible=True,
        )

    if cfg.outputs.html.carrier_revenue_distribution:
        rpt.add_figure(summary.fig_carrier_revenue_distribution(also_df=True))

    if cfg.outputs.html.leg_load_factor_distribution:
        rpt.add_figure(summary.fig_leg_load_factor_distribution(also_df=True))

    if cfg.outputs.html.carrier_local_share:
        rpt.add_figure(summary.fig_carrier_local_share(also_df=True))
        rpt.add_figure(summary.fig_carrier_local_share("leg_pax", also_df=True))

    # bid price history is suppressed unless there is some bid price data
    if cfg.outputs.html.bid_price_history:
        try:
            bph = summary.bid_price_history
        except AttributeError:
            bph = None
        if bph is not None and bph["bid_price_mean"].max() > 0:
            rpt.add_figure(summary.fig_bid_price_history(also_df=True))

    # displacement history is suppressed unless there is some displacement data
    if cfg.outputs.html.displacement_history:
        try:
            dh = summary.displacement_history
        except AttributeError:
            dh = None
        if dh is not None and dh["displacement_mean"].max() > 0:
            rpt.add_figure(summary.fig_displacement_history(also_df=True))

    if extra is not None:
        rpt.add_extra(summary, *extra)

    if cfg.outputs.html.configs:
        rpt.new_section("Configuration")
        cfg_data = cfg.model_dump()
        for item in cfg.outputs.html.configs:
            if item in {"raw_license_certificate"}:
                # never include these in the HTML report
                continue
            if item not in cfg_data:
                rpt.new_section(item, level=2)
                rpt.current_section << Elem.from_string("<pre>Not available</pre>")
                continue
            rpt.new_section(f"{item}:", level=2)
            out = yaml.safe_dump(cfg_data[item]).replace("\n", "\n  ")
            rpt.current_section << Elem.from_string(f"<pre>  {out}</pre>")

    if cfg.outputs.html.metadata:
        rpt.new_section("Metadata")
        metadata_df = pd.Series(summary._metadata, name="value").rename_axis(index="key").to_frame().reset_index()
        rpt.current_section.append(metadata_df)

    if cfg.outputs.html.other:
        raise NotImplementedError("Other HTML sections not yet implemented")

    filename = filename_with_timestamp(filename, suffix=".html", make_dirs=make_dirs)

    return rpt.write(str(filename))
