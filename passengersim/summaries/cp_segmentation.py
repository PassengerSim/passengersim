from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd

from passengersim.reporting import report_figure
from passengersim.utils.nested_dict import from_nested_dict  # noqa: F401

from .generic import (
    GenericSimulationTables,
    SimulationTableItem,
)
from .tools import aggregate_by_summing_dataframe

if TYPE_CHECKING:
    import altair as alt

    from passengersim import Simulation


def new_tally() -> dict:
    """Create a new tally dictionary with default values."""
    return {
        "sold": 0.0,
        "revenue": 0.0,
        "cp_sold": 0.0,
        "cp_revenue": 0.0,
    }


def extract_cp_segmentation(sim: Simulation) -> pd.DataFrame:
    """Extract carrier-level summary data from a Simulation."""

    cp_data = defaultdict(lambda: defaultdict(new_tally))
    for fare in sim.sim.fares:
        rec = cp_data[fare.carrier.name][fare.booking_class]
        rec["sold"] += fare.gt_sold
        rec["revenue"] += fare.gt_revenue
        rec["cp_sold"] += fare.gt_cp_sold
        rec["cp_revenue"] += fare.gt_cp_revenue
    return from_nested_dict(cp_data, ["carrier", "booking_class", "measure"])


class SimTabContinuousPricingSegmentation(GenericSimulationTables):
    cp_segmentation: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_by_summing_dataframe("cp_segmentation"),
        extraction_func=extract_cp_segmentation,
        doc="Continuous pricing segmentation summary data.",
    )

    @report_figure
    def fig_cp_segmentation(
        self, *, raw_df: bool = False, also_df: bool = False
    ) -> alt.Chart | tuple[alt.Chart, pd.DataFrame] | pd.DataFrame:
        import altair as alt

        df = self.cp_segmentation.reset_index()[["carrier", "booking_class", "sold", "cp_sold"]]
        if raw_df:
            return df
        df["Zero"] = 0

        base_chart = alt.Chart(df)

        chart = (
            base_chart.mark_bar()
            .encode(
                y=alt.Y("booking_class:N", sort=None, title="Booking Class"),
                x=alt.X("sold:Q", title="Total Sold", stack=False, axis=alt.Axis()),
                color=alt.Color("booking_class:N", sort=None, title="Booking Class"),
                tooltip=[
                    alt.Tooltip("carrier:N", title="Carrier"),
                    alt.Tooltip("booking_class:N", title="Booking Class"),
                    alt.Tooltip("sold:Q", title="Total Sold", format=","),
                    alt.Tooltip("cp_sold:Q", title="Continuous Priced Sold", format=","),
                ],
            )
            .properties(width=300, height=500)
        )

        # Create a second chart for cp_sold, overlaid as error bars on the first
        chart_cp = (
            base_chart.transform_filter(
                alt.datum.cp_sold > 0  # Only show nonzero counts for cp_sold
            )
            .mark_errorbar(extent="ci", ticks=True)
            .encode(
                y=alt.Y("booking_class:N", sort=None, title="Booking Class"),
                x=alt.X("cp_sold:Q", title="Continuous Priced Sold"),
                x2=alt.X2("Zero:Q", title="Zero"),
                color=alt.value("black"),
                tooltip=[
                    alt.Tooltip("carrier:N", title="Carrier"),
                    alt.Tooltip("booking_class:N", title="Booking Class"),
                    alt.Tooltip("sold:Q", title="Total Sold", format=","),
                    alt.Tooltip("cp_sold:Q", title="Continuous Priced Sold", format=","),
                ],
                strokeWidth=alt.value(2),  # Set stroke width for the error bars
            )
        )

        fig = (chart + chart_cp).facet(
            facet=alt.Facet("carrier:N", title="Carrier"),
        )

        if also_df:
            return fig, df
        else:
            return fig
