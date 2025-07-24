from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from .generic import GenericSimulationTables, SimulationTableItem
from .pathclasses import SimTabPathClasses
from .tools import aggregate_by_first_dataframe

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from passengersim import Simulation


def extract_path_legs(sim: Simulation) -> pd.DataFrame:
    """
    Extract path_legs from a Simulation.

    This is a table that indicates which legs are included in each path.

    Parameters
    ----------
    sim : Simulation
        The Simulation object from which to extract the data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns "path_id" and "leg_id".
    """
    path_legs = []
    for pth in sim.sim.paths:
        for leg_id in pth.leg_ids:
            path_legs.append((pth.path_id, leg_id))
    path_legs = pd.DataFrame(path_legs, columns=["path_id", "leg_id"])
    return path_legs


class SimTabPathLegs(GenericSimulationTables):
    """Container for summary tables and figures extracted from a Simulation.

    This class is a subclass of GenericSimulationTables, which is defined in
    the generic module.  It lists the items that are available in the
    SimulationTables class, and provides type hints and (optionally, but
    ideally) documentation for the data that is stored in each item.
    """

    path_legs: pd.DataFrame = SimulationTableItem(
        aggregation_func=aggregate_by_first_dataframe("path_legs"),
        extraction_func=extract_path_legs,
        doc="Legs on each path.",
    )

    pathclasses: pd.DataFrame = SimTabPathClasses.pathclasses

    def select_leg_analysis(self, leg_id: int | ArrayLike[int]) -> dict[str, pd.DataFrame]:
        """
        Select path_legs for a specific leg.

        Parameters
        ----------
        leg_id : int
            The leg_id(s) to select.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys include "orig", "dest", and "booking_class".  Values
            are DataFrames with columns "gt_sold" and "gt_revenue".
        """
        if isinstance(leg_id, int):
            path_ids = self.path_legs.path_id[self.path_legs.leg_id == leg_id]
        else:
            path_ids = self.path_legs.path_id[self.path_legs.leg_id.isin(leg_id)]
        which_path = self.pathclasses.index.get_level_values("path_id").isin(path_ids)
        df = self.pathclasses.loc[which_path]
        result = {}
        for k in ["orig", "dest", "booking_class"]:
            result[k] = df.groupby(k)[["gt_sold", "gt_revenue"]].sum()
            result[k] = result[k].query("gt_sold > 0")
        return result

    def fig_select_leg_analysis(
        self,
        leg_id: int | ArrayLike[int],
        metric: Literal["bookings", "revenue"] = "bookings",
        *,
        raw_input: dict[str, pd.DataFrame] = None,
    ):
        """
        Select path_legs for a specific leg.

        Parameters
        ----------
        leg_id : int
            The leg_id(s) to select.
        metric : str
            The metric to display.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns "path_id" and "leg_id".
        """
        if isinstance(raw_input, dict):
            data = raw_input
        else:
            data = self.select_leg_analysis(leg_id)

        if isinstance(leg_id, int):
            leg_descrip = f"Leg Id {leg_id}"
        elif len(leg_id) == 1:
            leg_descrip = f"Leg Id {leg_id[0]}"
        elif len(leg_id) > 4:
            leg_descrip = f"{len(leg_id)} Selected Leg Ids"
        else:
            leg_descrip = f"Leg Ids {list(leg_id)}"

        import altair as alt

        if metric == "bookings":
            x = alt.X("gt_sold")
        elif metric == "revenue":
            x = alt.X("gt_revenue")
        else:
            raise ValueError(f"Unknown metric: {metric}")

        charts = []
        for k in ["orig", "dest"]:
            df = data[k]
            # for k, df in data.items():
            chart = (
                alt.Chart(df.reset_index())
                .mark_bar()
                .encode(
                    x=x.title(k.replace("_", " ").title()),
                    color=alt.Color(k),
                    tooltip=[
                        alt.Tooltip(k, title=k.replace("_", " ").title()),
                        alt.Tooltip("gt_sold", title="Bookings", format=".4s"),
                        alt.Tooltip("gt_revenue", title="Revenue", format=".4s"),
                    ],
                )
            )
            charts.append(chart)

        orig_dest_chart = (
            alt.vconcat(*charts)
            # .resolve_scale(color="independent")
            # .properties(
            #     title={
            #         "text": [f"{metric.title()} on {leg_descrip}"],
            #     }
            # )
        )

        booking_class_chart = (
            alt.Chart(data["booking_class"].reset_index())
            .mark_bar()
            .encode(
                x=x.title("Booking Class"),
                color=alt.Color(
                    "booking_class",
                    # legend=alt.Legend(orient="bottom")
                ),
                tooltip=[
                    alt.Tooltip("booking_class", title="Booking Class"),
                    alt.Tooltip("gt_sold", title="Bookings", format=".4s"),
                    alt.Tooltip("gt_revenue", title="Revenue", format=".4s"),
                ],
            )
        )

        try:
            return (
                alt.hconcat(orig_dest_chart, booking_class_chart)
                .resolve_scale(color="independent")
                .properties(
                    title={
                        "text": [f"{metric.title()} on {leg_descrip}"],
                    }
                )
            )
        except Exception as e:
            import sys

            print(e, file=sys.stderr)
            return [orig_dest_chart, booking_class_chart]
