"""Visualization tools for comparing exactly two simulations.

Each function here takes two positional arguments that each provide a `SimulationTables` object,
the "baseline" and "treatment" scenarios to compare. The functions then produce visualizations
that show the shift performance metrics (e.g. load factor, local share) from the baseline
to the treatment scenarios.
"""

from typing import Literal

import altair as alt
import pandas as pd

from passengersim.summaries import SimulationTables


def fig_leg_shifts(
    baseline: SimulationTables,
    treatment: SimulationTables,
    *,
    coloring: Literal["avg_revenue", "change_revenue", "change_revenue_pct", None] = "avg_revenue",
    size_range: tuple[int, int] = (10, 100),
    facet_width: int = 400,
    facet_height: int = 400,
    facet_columns: int = 2,
    opacity: float = 1.0,
    fillOpacity: float = 1.0,
    strokeOpacity: float = 1.0,
    strokeWidth: float = 1.0,
    raw_df: bool = False,
):
    """Create a scatter/shift chart comparing leg-level metrics between two simulations.

    For each leg, a "lollipop" is drawn, with the stick showing the change from the
    baseline to the treatment position in load-factor vs. local-share space, and the
    point marking the treatment result. The size of the point reflects the leg total
    capacity (across all cabins). This makes it easy to see both the magnitude
    and direction of change for every leg.

    Parameters
    ----------
    baseline : SimulationTables
        Simulation results to use as the reference (baseline) scenario.
    treatment : SimulationTables
        Simulation results to compare against the baseline (treatment) scenario.
    coloring : {"avg_revenue", "change_revenue", "change_revenue_pct"}, optional
        Determines how points and lines are colored:

        - ``"avg_revenue"`` – color by absolute average revenue in the treatment.
        - ``"change_revenue"`` – color by the absolute change in average revenue
          (treatment minus baseline), using a diverging red-blue scale centered at 0.
        - ``"change_revenue_pct"`` – color by the percentage change in average revenue,
          using a diverging red-blue scale centered at 0.
        - ``None`` – color by carrier (nominal color encoding).
    size_range : tuple[int, int], optional
        Minimum and maximum point sizes (in pixels²) used to encode leg capacity.
    facet_width : int, optional
        Width of each facet panel in pixels.
    facet_height : int, optional
        Height of each facet panel in pixels.
    facet_columns : int, optional
        Number of columns in the faceted layout (one facet per carrier).
    opacity : float, optional
        Overall opacity of the point marks.
    fillOpacity : float, optional
        Fill opacity of the point marks.
    strokeOpacity : float, optional
        Stroke opacity of both point and line marks.
    strokeWidth : float, optional
        Stroke width of the line (rule) marks.

    Returns
    -------
    altair.FacetChart
        An interactive Altair faceted chart (faceted by carrier) showing the shift
        in load factor and local share for each leg between baseline and treatment.
    """
    df_baseline = baseline.legs_
    df_treatment = treatment.legs
    df_baseline["avg_revenue"] = df_baseline["gt_revenue"] / baseline.n_total_samples
    df_treatment["avg_revenue"] = df_treatment["gt_revenue"] / treatment.n_total_samples
    df = pd.merge(
        df_baseline,
        df_treatment,
        on=["leg_id", "carrier", "orig", "dest", "flt_no", "distance"],
        suffixes=("_baseline", "_treatment"),
    )
    df["avg_revenue_change"] = df["avg_revenue_treatment"] - df["avg_revenue_baseline"]
    df["avg_revenue_change_pct"] = df["avg_revenue_change"] / df["avg_revenue_baseline"]

    if raw_df:
        return df

    chart = alt.Chart(df.reset_index())

    if coloring == "avg_revenue":
        color = alt.Color(
            "avg_revenue_treatment:Q",
            title="Avg Revenue",
        )
    elif coloring == "change_revenue":
        color = alt.Color(
            "avg_revenue_change:Q",
            title="Δ Avg Revenue",
            scale=alt.Scale(
                scheme="redblue",  # Use a diverging scheme
                domainMid=0,  # Forces the center of the scheme to 0
            ),
        )
    elif coloring == "change_revenue_pct":
        color = alt.Color(
            "avg_revenue_change_pct:Q",
            title="Δ Avg Revenue %",
            scale=alt.Scale(
                scheme="redblue",  # Use a diverging scheme
                domainMid=0,  # Forces the center of the scheme to 0
            ),
            legend=alt.Legend(format=".1%"),
        )
    elif coloring is None:
        color = alt.Color("carrier:N", title="Carrier")
    else:
        raise ValueError(f"Unknown coloring {coloring}")

    tooltips = [
        alt.Tooltip("carrier", title="Carrier"),
        alt.Tooltip("orig", title="Origin"),
        alt.Tooltip("dest", title="Destination"),
        alt.Tooltip("avg_load_factor_baseline", title="Avg Load Factor (Baseline)", format=".2f"),
        alt.Tooltip("avg_load_factor_treatment", title="Avg Load Factor (Treatment)", format=".2f"),
        alt.Tooltip("avg_local_baseline", title="Avg Local Share (Baseline)", format=".2f"),
        alt.Tooltip("avg_local_treatment", title="Avg Local Share (Treatment)", format=".2f"),
        alt.Tooltip("avg_revenue_baseline", title="Avg Revenue (Baseline)", format="$.3s"),
        alt.Tooltip("avg_revenue_treatment", title="Avg Revenue (Treatment)", format="$.3s"),
        alt.Tooltip("avg_revenue_change", title="Δ Avg Revenue", format="$.3s"),
        alt.Tooltip("avg_revenue_change_pct", title="Δ Avg Revenue %", format=".2%"),
        alt.Tooltip("capacity"),
    ]

    lines = chart.mark_rule(strokeWidth=strokeWidth, strokeOpacity=strokeOpacity).encode(
        x=alt.X("avg_load_factor_baseline", scale=alt.Scale(zero=False), title="Avg Load Factor"),
        x2="avg_load_factor_treatment",
        y=alt.Y("avg_local_baseline", scale=alt.Scale(zero=False), title="Avg Local Share"),
        y2=alt.Y2("avg_local_treatment"),
        color=color,
        tooltip=tooltips,
    )

    points = chart.mark_point(
        filled=True, opacity=opacity, fillOpacity=fillOpacity, strokeOpacity=strokeOpacity
    ).encode(
        x=alt.X("avg_load_factor_treatment", scale=alt.Scale(zero=False), title="Avg Load Factor"),
        y=alt.Y("avg_local_treatment", scale=alt.Scale(zero=False), title="Avg Local Share"),
        size=alt.Size("capacity:Q", title="Capacity", scale=alt.Scale(range=size_range)),
        color=color,
        tooltip=tooltips,
    )

    return (
        (points + lines)
        .properties(width=facet_width, height=facet_height)
        .facet("carrier", columns=facet_columns)
        .interactive()
    )


def fig_service_shifts(
    baseline: SimulationTables,
    treatment: SimulationTables,
    *,
    coloring: Literal["avg_revenue", "change_revenue", None] = "avg_revenue",
    size_range: tuple[int, int] = (10, 100),
    facet_width: int = 400,
    facet_height: int = 400,
    facet_columns: int = 2,
    opacity: float = 1.0,
    fillOpacity: float = 1.0,
    strokeOpacity: float = 1.0,
    strokeWidth: float = 1.0,
):
    """Create a scatter/shift chart comparing service-level metrics between two simulations.

    For each O&D service, a "lollipop" is drawn, with the stick showing the shift from
    the baseline to the treatment position in load-factor vs. local-share space, and the
    point marking the treatment result.  Services are the aggregation of all legs sharing
    a common carrier, origin, and destination.

    Parameters
    ----------
    baseline : SimulationTables
        Simulation results to use as the reference (baseline) scenario.
    treatment : SimulationTables
        Simulation results to compare against the baseline (treatment) scenario.
    coloring : {"avg_revenue", "change_revenue"}, optional
        Determines how points and lines are colored:

        - ``"avg_revenue"`` – color by absolute average revenue in the treatment.
        - ``"change_revenue"`` – color by the absolute change in average revenue
          (treatment minus baseline), using a diverging red-blue scale centered at 0.
        - ``None`` – color by carrier (nominal color encoding).
    size_range : tuple[int, int], optional
        Minimum and maximum point sizes (in pixels²) used to encode service capacity.
    facet_width : int, optional
        Width of each facet panel in pixels.
    facet_height : int, optional
        Height of each facet panel in pixels.
    facet_columns : int, optional
        Number of columns in the faceted layout (one facet per carrier).
    opacity : float, optional
        Overall opacity of the point marks.
    fillOpacity : float, optional
        Fill opacity of the point marks.
    strokeOpacity : float, optional
        Stroke opacity of both point and line marks.
    strokeWidth : float, optional
        Stroke width of the line (rule) marks.

    Returns
    -------
    altair.FacetChart
        An interactive Altair faceted chart (faceted by carrier) showing the shift
        in load factor and local share for each service between baseline and treatment.
    """
    df_baseline = baseline.services
    df_treatment = treatment.services
    df = pd.merge(
        df_baseline,
        df_treatment,
        on=["carrier", "orig", "dest", "distance", "capacity", "frequency"],
        suffixes=("_baseline", "_treatment"),
    )
    df["avg_revenue_change"] = df["avg_revenue_treatment"] - df["avg_revenue_baseline"]

    chart = alt.Chart(df.reset_index())

    if coloring == "avg_revenue":
        color = alt.Color(
            "avg_revenue_treatment:Q",
            title="Avg Revenue",
        )
    elif coloring == "change_revenue":
        color = alt.Color(
            "avg_revenue_change:Q",
            title="Δ Avg Revenue",
            scale=alt.Scale(
                scheme="redblue",  # Use a diverging scheme
                domainMid=0,  # Forces the center of the scheme to 0
            ),
        )
    elif coloring is None:
        color = alt.Color("carrier:N", title="Carrier")
    else:
        raise ValueError(f"Unknown coloring {coloring}")

    tooltips = [
        alt.Tooltip("carrier", title="Carrier"),
        alt.Tooltip("orig", title="Origin"),
        alt.Tooltip("dest", title="Destination"),
        alt.Tooltip("avg_load_factor_baseline", title="Avg Load Factor (Baseline)", format=".2f"),
        alt.Tooltip("avg_load_factor_treatment", title="Avg Load Factor (Treatment)", format=".2f"),
        alt.Tooltip("avg_local_baseline", title="Avg Local Share (Baseline)", format=".2f"),
        alt.Tooltip("avg_local_treatment", title="Avg Local Share (Treatment)", format=".2f"),
        alt.Tooltip("avg_revenue_baseline", title="Avg Revenue (Baseline)", format="$.3s"),
        alt.Tooltip("avg_revenue_treatment", title="Avg Revenue (Treatment)", format="$.3s"),
        alt.Tooltip("avg_revenue_change", title="Δ Avg Revenue", format="$.3s"),
        alt.Tooltip("capacity"),
    ]

    lines = chart.mark_rule(strokeWidth=strokeWidth, strokeOpacity=strokeOpacity).encode(
        x=alt.X("avg_load_factor_baseline", scale=alt.Scale(zero=False), title="Avg Load Factor"),
        x2="avg_load_factor_treatment",
        y=alt.Y("avg_local_baseline", scale=alt.Scale(zero=False), title="Avg Local Share"),
        y2=alt.Y2("avg_local_treatment"),
        color=color,
        tooltip=tooltips,
    )

    points = chart.mark_point(
        filled=True, opacity=opacity, fillOpacity=fillOpacity, strokeOpacity=strokeOpacity
    ).encode(
        x=alt.X("avg_load_factor_treatment", scale=alt.Scale(zero=False), title="Avg Load Factor"),
        y=alt.Y("avg_local_treatment", scale=alt.Scale(zero=False), title="Avg Local Share"),
        size=alt.Size("capacity:Q", title="Capacity", scale=alt.Scale(range=size_range)),
        color=color,
        tooltip=tooltips,
    )

    return (
        (points + lines)
        .properties(width=facet_width, height=facet_height)
        .facet("carrier", columns=facet_columns)
        .interactive()
    )
