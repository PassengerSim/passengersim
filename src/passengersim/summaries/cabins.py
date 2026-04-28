#
# Data for analyzing multi-cabin
#
# AlanW, October 2025
# (C) PassengerSIM LLC
#

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from .generic import (
    GenericSimulationTables,
    SimulationTableItem,
)
from .tools import aggregate_by_concat_dataframe

if TYPE_CHECKING:
    import altair as alt

    from passengersim import Simulation


def extract_cabins(sim: Simulation) -> pd.DataFrame | None:
    """Extract cabin data from the Cabin class.
    Used internally to create the output summary"""
    combined_data = []
    for leg in sim.eng.legs:
        for cab in leg.cabins:
            data = cab.get_cabin_data(leg.carrier_name, leg.flt_no)
            combined_data += data
    if len(combined_data) == 0:
        return None
    df = pd.DataFrame.from_dict(combined_data)
    df = df.set_index(["trial", "sample", "carrier", "flt_no", "cabin"])
    return df


class SimTabCabins(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    cabins: pd.DataFrame | None = SimulationTableItem(
        aggregation_func=aggregate_by_concat_dataframe("cabins"),
        extraction_func=extract_cabins,
        doc="Cabin-level summary data from each sample",
    )

    def fig_cabin_load_factors(self, raw_df: bool = False) -> alt.Chart | pd.DataFrame:
        import altair as alt

        all_cabins_df = self.get_cabin_df(self)
        if raw_df:
            return all_cabins_df

        fig = alt.vconcat()
        for cabin in ["C", "Y"]:
            cabin_df = all_cabins_df[all_cabins_df["cabin"] == cabin]
            base = alt.Chart(cabin_df).encode(
                y=alt.Y("experiment:N", title=""),
                x=alt.X(
                    "load_factor:Q", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1]), title="Load Factor"
                ),
                color="experiment:N",
                tooltip=alt.Tooltip("load_factor", format=".1%"),
                text=alt.Text("load_factor:Q", format=".1%"),
            )

            z = alt.layer(
                base.mark_bar(),
                base.mark_text(color="black", baseline="top", dx=20, dy=-5),
            ).facet(row=alt.Row("carrier:N", title=""), title=alt.TitleParams(text=f"Cabin: {cabin}", anchor="middle"))
            fig &= z
        return fig

    def get_cabins_df(_summaries):
        """Get all the cabine data into a dataframe"""
        tmp_df = None
        for k, exp_df in _summaries.items():
            df2 = exp_df.cabins.copy(deep=True)
            df2["experiment"] = k
            if tmp_df is None:
                tmp_df = df2
            else:
                tmp_df = pd.concat([tmp_df, df2])

        tmp_df["asm"] = tmp_df["distance"] * tmp_df["capacity"]
        tmp_df["rpm"] = tmp_df["distance"] * tmp_df["sold"]
        cabin_df = (
            tmp_df.groupby(["experiment", "carrier", "cabin"])
            .agg({"asm": "sum", "rpm": "sum", "sold": "sum", "revenue": "sum"})
            .reset_index()
        )
        cabin_df["load_factor"] = cabin_df["rpm"] / cabin_df["asm"]
        cabin_df["sold"] /= 600
        cabin_df["revenue"] /= 600
        return cabin_df
