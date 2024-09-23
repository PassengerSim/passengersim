from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import pandas as pd

from passengersim.reporting import report_figure

from .generic import (
    SimulationTableItem,
    _GenericSimulationTables,
)

if TYPE_CHECKING:
    from passengersim import Simulation

    from . import SimulationTables


def extract_carriers(sim: Simulation) -> pd.DataFrame:
    """Extract carrier-level summary data from a Simulation."""
    eng = sim.sim
    num_samples = eng.num_trials_completed * (eng.num_samples - eng.burn_samples)

    carrier_asm = defaultdict(float)
    carrier_rpm = defaultdict(float)
    carrier_leg_lf = defaultdict(float)
    carrier_leg_count = defaultdict(float)
    for leg in eng.legs:
        carrier_name = leg.carrier.name
        carrier_asm[carrier_name] += leg.distance * leg.capacity * num_samples
        carrier_rpm[carrier_name] += leg.distance * leg.gt_sold
        carrier_leg_lf[carrier_name] += leg.gt_sold / (leg.capacity * num_samples)
        carrier_leg_count[carrier_name] += 1

    carrier_data = []
    for carrier in sim.sim.carriers:
        avg_rev = carrier.gt_revenue / num_samples
        rpm = carrier_rpm[carrier.name] / num_samples
        avg_leg_lf = (
            100 * carrier_leg_lf[carrier.name] / max(carrier_leg_count[carrier.name], 1)
        )
        # Add up total ancillaries
        tot_anc_rev = 0.0
        for anc in carrier.ancillaries:
            tot_anc_rev += anc.price * anc.sold
        carrier_data.append(
            {
                "carrier": carrier.name,
                "control": carrier.control,
                "avg_rev": avg_rev,
                "avg_sold": carrier.gt_sold / num_samples,
                "truncation_rule": carrier.truncation_rule,
                "avg_leg_lf": avg_leg_lf,
                "asm": carrier_asm[carrier.name] / num_samples,
                "rpm": rpm,
                "ancillary_rev": tot_anc_rev,
            }
        )
    return pd.DataFrame(carrier_data).set_index("carrier")


def aggregate_carriers(summaries: list[SimulationTables]) -> pd.DataFrame | None:
    """Aggregate leg-level summaries."""
    table_avg = []
    for s in summaries:
        frame = s._raw_carriers
        if frame is not None:
            table_avg.append(
                frame.set_index(["control", "truncation_rule"], append=True)
            )
    n = len(table_avg)
    while len(table_avg) > 1:
        table_avg[0] = table_avg[0].add(table_avg.pop(1), fill_value=0)
    if table_avg:
        table_avg[0] /= n
        return table_avg[0].reset_index(["control", "truncation_rule"])
    return None


class SimTabCarriers(_GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of _GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    carriers: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_carriers,
        extraction_func=extract_carriers,
        computed_fields={
            "avg_price": "avg_rev / avg_sold",
            "yield": "avg_rev / rpm",
            "sys_lf": "100.0 * rpm / asm",
        },
        doc="Carrier-level summary data.",
    )

    def _fig_carrier_attribute(
        self,
        raw_df: bool,
        load_measure: str,
        measure_name: str,
        measure_format: str = ".2f",
        orient: Literal["h", "v"] = "h",
        title: str | None = None,
    ):
        df = self.carriers.reset_index()[["carrier", load_measure]]
        if raw_df:
            return df
        import altair as alt

        chart = alt.Chart(df)
        if orient == "v":
            bars = chart.mark_bar().encode(
                x=alt.X("carrier:N", title="Carrier"),
                y=alt.Y(f"{load_measure}:Q", title=measure_name).stack("zero"),
                tooltip=[
                    alt.Tooltip("carrier", title="Carrier"),
                    alt.Tooltip(
                        f"{load_measure}:Q", title=measure_name, format=measure_format
                    ),
                ],
            )
            text = chart.mark_text(dx=0, dy=3, color="white", baseline="top").encode(
                x=alt.X("carrier:N", title="Carrier"),
                y=alt.Y(f"{load_measure}:Q", title=measure_name).stack("zero"),
                text=alt.Text(f"{load_measure}:Q", format=measure_format),
            )
        else:
            bars = chart.mark_bar().encode(
                y=alt.Y("carrier:N", title="Carrier"),
                x=alt.X(f"{load_measure}:Q", title=measure_name).stack("zero"),
                tooltip=[
                    alt.Tooltip("carrier", title="Carrier"),
                    alt.Tooltip(
                        f"{load_measure}:Q", title=measure_name, format=measure_format
                    ),
                ],
            )
            text = chart.mark_text(
                dx=-5, dy=0, color="white", baseline="middle", align="right"
            ).encode(
                y=alt.Y("carrier:N", title="Carrier"),
                x=alt.X(f"{load_measure}:Q", title=measure_name).stack("zero"),
                text=alt.Text(f"{load_measure}:Q", format=measure_format),
            )
        fig = (
            (bars + text)
            .properties(
                width=500,
                height=10 + 20 * len(df),
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
        if title:
            fig.title = title
        return fig

    @report_figure
    def fig_carrier_load_factors(
        self,
        load_measure: Literal["sys_lf", "avg_leg_lf"] = "sys_lf",
        *,
        raw_df=False,
    ):
        measure_name = (
            "System Load Factor" if load_measure == "sys_lf" else "Leg Load Factor"
        )
        return self._fig_carrier_attribute(
            raw_df,
            load_measure,
            measure_name,
            title=f"Carrier {measure_name}s",
        )

    @report_figure
    def fig_carrier_revenues(self, *, raw_df=False):
        return self._fig_carrier_attribute(
            raw_df, "avg_rev", "Average Revenue", "$.4s", title="Carrier Revenues"
        )

    @report_figure
    def fig_carrier_yields(self, *, raw_df=False):
        return self._fig_carrier_attribute(
            raw_df, "yield", "Average Yield", "$.4f", title="Carrier Yields"
        )

    @report_figure
    def fig_carrier_total_bookings(self: SimulationTables, *, raw_df=False):
        return self._fig_carrier_attribute(
            raw_df,
            "avg_sold",
            "Total Bookings",
            ".4s",
            title="Carrier Total Bookings",
        )

    @report_figure
    def fig_carrier_mileage(self, *, raw_df: bool = False):
        """
        Figure showing mileage by carrier.

        ASM is available seat miles, and RPM is revenue passenger miles. Both
        measures are reported as the average across all non-burned samples.

        Parameters
        ----------
        raw_df : bool, default False
            Return the raw data for this figure as a pandas DataFrame, instead
            of generating the figure itself.
        report : xmle.Reporter, optional
            Also append this figure to the given report.
        trace : pd.ExcelWriter, optional
            Also write the data from this figure to the given Excel file.
        """
        df = (
            self.carriers.reset_index()[["carrier", "asm", "rpm"]]
            .set_index("carrier")
            .rename_axis(columns="measure")
            .unstack()
            .to_frame("value")
            .reset_index()
        )
        if raw_df:
            return df
        import altair as alt

        chart = alt.Chart(df, title="Carrier Loads")
        bars = chart.mark_bar().encode(
            x=alt.X("carrier:N", title="Carrier"),
            y=alt.Y("value", stack=None, title="miles"),
            color="measure",
            tooltip=["carrier", "measure", alt.Tooltip("value", format=".4s")],
        )
        text = chart.mark_text(
            dx=0,
            dy=5,
            color="white",
            baseline="top",
        ).encode(
            x=alt.X("carrier:N"),
            y=alt.Y("value").stack(None),
            text=alt.Text("value:Q", format=".4s"),
        )
        fig = (
            (bars + text)
            .properties(
                width=400,
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
        return fig
