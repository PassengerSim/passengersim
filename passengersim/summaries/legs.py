from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from passengersim.reporting import report_figure

from .generic import GenericSimulationTables, SimulationTableItem
from .tools import aggregate_by_summing_dataframe, break_on_integer

if TYPE_CHECKING:
    from collections.abc import Collection

    from passengersim import Simulation


def extract_legs(sim: Simulation) -> pd.DataFrame:
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

    @report_figure
    def fig_load_factor_distribution(
        self,
        by_carrier: bool | str = True,
        breakpoints: Collection[int] = None,
        *,
        raw_df=False,
    ):
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
        raw_df : bool, default False
            Return the raw data for this figure as a pandas DataFrame, instead
            of generating the figure itself.

        Returns
        -------
        altair.Chart or pd.DataFrame
        """
        if breakpoints is None:
            breakpoints = range(25, 100, 5)  # default breakpoints

        title = "Load Factor Frequency"  # default title

        df_for_chart = (
            self.legs.assign(
                leg_load_factor_range=break_on_integer(
                    self.legs["avg_load_factor"],
                    breakpoints,
                    result_name="leg_load_factor_range",
                )
            )
            .groupby(["carrier", "leg_load_factor_range"], observed=False)
            .size()
            .rename("frequency")
            .reset_index()
        )

        if not by_carrier:
            df_for_chart = (
                df_for_chart.groupby(["leg_load_factor_range"], observed=False)
                .frequency.sum()
                .reset_index()
            )
        elif isinstance(by_carrier, str):
            df_for_chart = df_for_chart[df_for_chart["carrier"] == by_carrier]
            df_for_chart = df_for_chart.drop(columns=["carrier"])

        if raw_df:
            return df_for_chart

        import altair as alt

        if by_carrier is True:
            chart = (
                alt.Chart(df_for_chart)
                .mark_bar()
                .encode(
                    x=alt.X("leg_load_factor_range", title="Load Factor Range"),
                    y=alt.Y("frequency:Q", title="Count"),
                    facet=alt.Facet("carrier:N", columns=2, title="Carrier"),
                    tooltip=[
                        alt.Tooltip("carrier", title="Carrier"),
                        alt.Tooltip("leg_load_factor_range", title="Load Factor Range"),
                        alt.Tooltip("frequency", title="Count"),
                    ],
                )
                .properties(width=300, height=250, title=f"{title} by Carrier")
            )
        else:
            chart = (
                alt.Chart(df_for_chart)
                .mark_bar()
                .encode(
                    x=alt.X("leg_load_factor_range", title="Load Factor Range"),
                    y=alt.Y("frequency:Q", title="Count"),
                    tooltip=[
                        alt.Tooltip("carrier", title="Carrier"),
                        alt.Tooltip("leg_load_factor_range", title="Load Factor Range"),
                        alt.Tooltip("frequency", title="Count"),
                    ],
                )
                .properties(
                    width=600,
                    height=400,
                    title=title if not by_carrier else f"{title} ({by_carrier})",
                )
            )

        return chart
