from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import pandas as pd

from passengersim.database import common_queries
from passengersim.reporting import report_figure

from .generic import (
    DatabaseTableItem,
    GenericSimulationTables,
    SimulationTableItem,
)
from .tools import aggregate_by_concat_dataframe

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
    carrier_local_leg_pax = defaultdict(float)
    carrier_total_leg_pax = defaultdict(float)
    for leg in eng.legs:
        carrier_name = leg.carrier.name
        carrier_asm[carrier_name] += leg.distance * leg.capacity * num_samples
        carrier_rpm[carrier_name] += leg.distance * leg.gt_sold
        carrier_leg_lf[carrier_name] += leg.gt_sold / (leg.capacity * num_samples)
        carrier_leg_count[carrier_name] += 1
        carrier_local_leg_pax[carrier_name] += leg.gt_sold_local
        carrier_total_leg_pax[carrier_name] += leg.gt_sold

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
        gt_cp_sold = carrier.gt_cp_sold
        gt_cp_revenue = carrier.gt_cp_revenue
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
                "avg_local_leg_pax": carrier_local_leg_pax[carrier.name] / num_samples,
                "avg_total_leg_pax": carrier_total_leg_pax[carrier.name] / num_samples,
                "cp_sold": gt_cp_sold / num_samples,
                "cp_revenue": gt_cp_revenue / num_samples,
            }
        )
    if len(carrier_data) == 0:
        return None
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


def extract_carrier_history2(sim: Simulation) -> pd.DataFrame | None:
    """Extract carrier_history from the Carrier class."""
    combined_data = []
    for cxr in sim.sim.carriers:
        hist = cxr.get_carrier_history()
        combined_data += hist
    if len(combined_data) == 0:
        return None
    df = pd.DataFrame.from_dict(combined_data)
    df = df.set_index(["trial", "sample", "carrier"])
    return df


def extract_forecast_accuracy(sim: Simulation) -> pd.DataFrame | None:
    """Extract forecast accuracy from the Carrier class."""
    combined_data = []
    for cxr in sim.sim.carriers:
        hist = cxr.get_forecast_accuracy()
        combined_data += hist
    if len(combined_data) == 0:
        return None
    df = pd.DataFrame.from_dict(combined_data)
    df = df.set_index(
        ["trial", "sample", "carrier", "booking_class", "timeframe"]
    ).reset_index()
    return df


class SimTabCarriers(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of GenericSimulationTables, which is defined in
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
            "rasm": "avg_rev / asm",
            "sys_lf": "100.0 * rpm / asm",
            "local_pct_leg_pax": "100.0 * avg_local_leg_pax / avg_total_leg_pax",
            "local_pct_bookings": "100.0 * avg_local_leg_pax / avg_sold",
        },
        doc="Carrier-level summary data.",
    )

    carrier_history: pd.DataFrame | None = DatabaseTableItem(
        aggregation_func=aggregate_by_concat_dataframe("carrier_history"),
        query_func=common_queries.carrier_history,
        doc="Carrier-level summary data from each sample.",
    )

    carrier_history2: pd.DataFrame | None = SimulationTableItem(
        aggregation_func=aggregate_by_concat_dataframe("carrier_history2"),
        extraction_func=extract_carrier_history2,
        doc="Carrier-level summary data from each sample, "
        "new version with counters in CoreCarrier.",
    )

    forecast_accuracy: pd.DataFrame | None = SimulationTableItem(
        aggregation_func=aggregate_by_concat_dataframe("forecast_accuracy"),
        extraction_func=extract_forecast_accuracy,
        doc="Summary of forecast history, based on UA's EDGAR approach",
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
                color=alt.Color("carrier:N", title="Carrier", legend=None),
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
                color=alt.Color("carrier:N", title="Carrier", legend=None),
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
    def fig_carrier_rasm(self, *, raw_df=False):
        return self._fig_carrier_attribute(
            raw_df,
            "rasm",
            "Revenue per Available Seat Mile",
            "$.4f",
            title="Carrier Revenue per Available Seat Mile (RASM)",
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
    def fig_carrier_local_share(
        self,
        load_measure: Literal["bookings", "leg_pax"] = "bookings",
        *,
        raw_df=False,
    ):
        measure_name = (
            "Local Percent of Bookings"
            if load_measure == "bookings"
            else "Local Percent of Leg Passengers"
        )
        m = "local_pct_bookings" if load_measure == "bookings" else "local_pct_leg_pax"
        return self._fig_carrier_attribute(
            raw_df,
            m,
            measure_name,
            title=f"Carrier {measure_name}",
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

    def fig_carrier_revenue_distribution(self, *, raw_df=False):
        """Figure showing the distribution of carrier revenues."""
        if raw_df:
            raise NotImplementedError("Raw data not available for this figure.")
        import altair as alt

        fig = (
            alt.Chart(self.carrier_history2.reset_index())
            .transform_density(
                "revenue",
                groupby=["carrier"],
                as_=["revenue", "density"],
            )
            .mark_area()
            .encode(
                x=alt.X("revenue:Q", axis=alt.Axis(title="Revenue", format="$.3s")),
                y=alt.Y("density:Q", title="Density", axis=alt.Axis(labels=False)),
                color="carrier:N",
            )
            .facet(
                "carrier:N",
                title="Revenue Distribution by Carrier",
            )
        )
        return fig

    def fig_carrier_head_to_head_revenue(
        self, x_carrier: str, y_carrier: str, *, raw_df=False
    ):
        import altair as alt

        df1 = self.carrier_history2.query(f"carrier == '{x_carrier}'")
        df2 = self.carrier_history2.query(f"carrier == '{y_carrier}'")

        df = pd.concat(
            [
                df1["revenue"] / df1["revenue"].mean(),
                df2["revenue"] / df2["revenue"].mean(),
            ]
        )
        rng = df.min(), df.max()
        df = df.unstack("carrier").reset_index()
        if raw_df:
            return df

        diag = (
            alt.Chart(pd.DataFrame({x_carrier: rng, "AL2": rng}))
            .mark_line(color="red", opacity=0.3)
            .encode(
                x=x_carrier,
                y="AL2",
            )
        )

        fig = (
            alt.Chart(df)
            .mark_circle(opacity=0.3)
            .encode(
                x=alt.X(f"{x_carrier}:Q")
                .axis(format="%")
                .scale(zero=False)
                .title(f"{x_carrier} Percentage of Mean Revenue"),
                y=alt.Y(f"{y_carrier}:Q")
                .axis(format="%")
                .scale(zero=False)
                .title(f"{y_carrier} Percentage of Mean Revenue"),
            )
            + diag
        )

        return fig.interactive()
