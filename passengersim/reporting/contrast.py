from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from passengersim.contrast import Contrast
from passengersim.types import PathLike
from passengersim.utils.bootstrap import BootstrapHtml

if TYPE_CHECKING:
    from passengersim import Config


def to_html(
    summaries: Contrast,
    filename: PathLike | None = None,
    *,
    base_config: Config | None = None,
    make_dirs: bool = True,
    title: str | None = None,
    carrier_revenues: bool = True,
    carrier_total_bookings: bool = True,
    carrier_load_factors: bool = True,
    carrier_yields: bool = True,
    carrier_rasm: bool = True,
    carrier_local_share: bool = True,
    fare_class_mix: bool = True,
    bookings_by_timeframe: bool = True,
    bid_price_history: bool = True,
    extra: tuple | None = None,
    frontmatter: tuple | None = None,
) -> pathlib.Path:
    """
    Write a summary to an HTML file.

    Parameters
    ----------
    summaries : Contrast
    filename : Path-like, optional
        If not provided, the filename will be taken from the run config.  If
        it is also not defined there, a ValueError will be raised.
    make_dirs : bool, optional
        If True, create any necessary directories.

    Returns
    -------

    """
    if filename is None:
        filename = summaries.config.outputs.html.filename
        if filename is None:
            raise ValueError("No filename provided")

    rpt = BootstrapHtml(title=title)

    rpt.new_section("Results")

    if carrier_revenues:
        rpt.add_figure(summaries.fig_carrier_revenues())

    if carrier_total_bookings:
        rpt.add_figure(summaries.fig_carrier_total_bookings())

    if carrier_load_factors:
        rpt.add_figure(summaries.fig_carrier_load_factors())

    if carrier_yields:
        rpt.add_figure(summaries.fig_carrier_yields())

    if carrier_rasm:
        rpt.add_figure(summaries.fig_carrier_rasm())

    if carrier_local_share:
        rpt.add_figure(summaries.fig_carrier_local_share(load_measure="bookings"))
        rpt.add_figure(summaries.fig_carrier_local_share(load_measure="leg_pax"))

    if fare_class_mix:
        rpt.add_figure(summaries.fig_fare_class_mix())

    if bid_price_history:
        rpt.add_figure(summaries.fig_bid_price_history())

    if bookings_by_timeframe:
        rpt.add_figure(summaries.fig_bookings_by_timeframe())
        if isinstance(bookings_by_timeframe, list | tuple):
            carriers = bookings_by_timeframe
        elif base_config is not None:
            carriers = list(base_config.carriers.keys())
        else:
            carriers = []
        for carrier in carriers:
            rpt.add_figure(summaries.fig_bookings_by_timeframe(by_carrier=carrier))
            rpt.add_figure(
                summaries.fig_bookings_by_timeframe(by_carrier=carrier, by_class=True)
            )

    if extra is not None:
        rpt.add_extra(summaries, *extra)

    if frontmatter is not None:
        if isinstance(frontmatter, str):
            frontmatter = (frontmatter,)
        rpt.add_frontmatter(*frontmatter)

    return rpt.write(filename, make_dirs=make_dirs)
