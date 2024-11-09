from __future__ import annotations

import pathlib
import time
from typing import TYPE_CHECKING

import yaml

from passengersim.config import Config

from .report import Elem, Report

if TYPE_CHECKING:
    from passengersim.summaries import SimulationTables


def to_html(
    summary: SimulationTables,
    filename: str | None = None,
    *,
    cfg: Config | None = None,
    make_dirs: bool = True,
) -> None:
    """
    Write a summary to an HTML file.

    Parameters
    ----------
    summary : SimulationTables
    filename : Path-like, optional
        If not provided, the filename will be taken from the run config.  If
        it is also not defined there, a ValueError will be raised.
    cfg : Config, optional
        If not provided, the configuration will be taken from the summary
    make_dirs : bool, optional
        If True, create any necessary directories.

    Returns
    -------

    """
    if cfg is None:
        cfg = summary.config
        if cfg is None:
            raise ValueError("No configuration provided")

    if filename is None:
        filename = pathlib.Path(cfg.outputs.html.filename)
        if filename is None:
            raise ValueError("No filename provided")

    rpt = Report(title=cfg.outputs.html.title or cfg.scenario)

    if cfg.outputs.html.carrier_revenues:
        rpt.add_section("Carrier Revenues")
        rpt << summary.fig_carrier_revenues()

    if cfg.outputs.html.carrier_load_factors:
        rpt.add_section("Carrier Load Factors")
        rpt << summary.fig_carrier_load_factors()

    if cfg.outputs.html.carrier_yields:
        rpt.add_section("Carrier Yields")
        rpt << summary.fig_carrier_yields()

    if cfg.outputs.html.carrier_table:
        rpt.add_section("Carrier Data")
        rpt << summary.carriers

    if cfg.outputs.html.fare_class_mix:
        rpt.add_section("Fare Class Mix")
        rpt << Elem.from_altair(summary.fig_fare_class_mix())

    if cfg.outputs.html.bookings_by_timeframe:
        if cfg.outputs.html.bookings_by_timeframe is True:
            carriers = list(cfg.carriers.keys())
        else:
            carriers = cfg.outputs.html.bookings_by_timeframe
        rpt.add_section("Bookings by Timeframe")
        for c in carriers:
            rpt << summary.fig_segmentation_by_timeframe(
                "bookings", by_carrier=c, by_class=True
            )

    if cfg.outputs.html.leg_load_factor_distribution:
        rpt.add_section("Leg Load Factor Distribution")
        rpt << summary.fig_leg_load_factor_distribution()

    # bid price history is suppressed unless there is some bid price data
    if cfg.outputs.html.bid_price_history:
        try:
            bph = summary.bid_price_history
        except AttributeError:
            bph = None
        if bph is not None and bph["bid_price_mean"].max() > 0:
            rpt.add_section("Bid Price History")
            rpt << summary.fig_bid_price_history()

    # displacement history is suppressed unless there is some displacement data
    if cfg.outputs.html.displacement_history:
        try:
            dh = summary.displacement_history
        except AttributeError:
            dh = None
        if dh is not None and dh["displacement_mean"].max() > 0:
            rpt.add_section("Displacement Cost History")
            rpt << summary.fig_displacement_history()

    if cfg.outputs.html.configs:
        rpt.add_section("Run Configuration")
        cfg_data = cfg.model_dump()
        for item in cfg.outputs.html.configs:
            if item in {"raw_license_certificate"}:
                # never include these in the HTML report
                continue
            if item not in cfg_data:
                rpt.add_section(item, level=2)
                rpt << Elem.from_string("<pre>Not available</pre>")
                continue
            rpt.add_section(item, level=2)
            out = yaml.safe_dump(cfg_data[item])
            rpt << Elem.from_string(f"<pre>{out}</pre>")

    if cfg.outputs.html.other:
        raise NotImplementedError("Other HTML sections not yet implemented")

    filename = pathlib.Path(filename)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = filename.with_suffix(f".{timestamp}.html")

    if make_dirs:
        filename.parent.mkdir(parents=True, exist_ok=True)

    rpt.save(str(filename))
