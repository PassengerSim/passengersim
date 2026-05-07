from typing import Literal

import altair as alt
import pandas as pd

from passengersim.summaries import SimulationTables


def fig_leg_shifts(
    baseline: SimulationTables,
    treatment: SimulationTables,
    *,
    coloring: Literal["avg_revenue", "change_revenue"] = "avg_revenue",
    size_range: tuple[int, int] = (10, 100),
    facet_width: int = 400,
    facet_height: int = 400,
    facet_columns: int = 2,
    opacity: float = 1.0,
    fillOpacity: float = 1.0,
    strokeOpacity: float = 1.0,
    strokeWidth: float = 1.0,
):
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


def fig_service_shifts(
    baseline: SimulationTables,
    treatment: SimulationTables,
    *,
    coloring: Literal["avg_revenue", "change_revenue"] = "avg_revenue",
    size_range: tuple[int, int] = (10, 100),
    facet_width: int = 400,
    facet_height: int = 400,
    facet_columns: int = 2,
    opacity: float = 1.0,
    fillOpacity: float = 1.0,
    strokeOpacity: float = 1.0,
    strokeWidth: float = 1.0,
):
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
