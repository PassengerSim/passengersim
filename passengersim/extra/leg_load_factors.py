import pandas as pd

from passengersim.config import Config


def leg_load_factors(leg_carried: pd.DataFrame, config: Config, passenger_count: str = "carried_all"):
    caps = {}
    carriers = {}
    for leg in config.legs:
        caps[leg.fltno] = leg.capacity
        carriers[leg.fltno] = leg.carrier
    df = leg_carried.groupby("flt_no")[[passenger_count]].sum()
    df["capacity"] = df.index.get_level_values("flt_no").map(caps).rename("capacity")
    df["carrier"] = df.index.get_level_values("flt_no").map(carriers).rename("carrier")
    df["lf"] = df[passenger_count] / df["capacity"] * 100
    df = df.reset_index().set_index(["flt_no", "carrier"])
    return df


def fig_leg_load_factors(
    x: pd.DataFrame,
    y: pd.DataFrame,
    x_label: str = "x",
    y_label: str = "y",
    config: Config | None = None,
):
    if x.legs is not None:
        df_x = x.legs.set_index(["flt_no", "carrier"])[["lf"]]
    else:
        df_x = leg_load_factors(x.leg_carried, config, passenger_count="carried_all")[["lf"]]

    if y.legs is not None:
        df_y = y.legs.set_index(["flt_no", "carrier"])[["lf"]]
    else:
        df_y = leg_load_factors(y.leg_carried, config, passenger_count="carried_all")[["lf"]]

    z = df_x.join(df_y, lsuffix="_x", rsuffix="_y").reset_index()

    import altair as alt

    carrier = "carrier"

    selection = alt.selection_point(fields=[carrier])
    color = alt.condition(
        selection,
        alt.Color(f"{carrier}:N", title="Carrier").legend(None),
        alt.value("lightgray"),
    )

    zz = z.reset_index()

    scatter = (
        alt.Chart(zz)
        .mark_point(clip=True)
        .encode(
            y=alt.Y("lf_y", title=f"{y_label} Load Factor").scale(domain=(25, 96)),
            x=alt.X("lf_x", title=f"{x_label} Load Factor").scale(domain=(25, 96)),
            color=color,
        )
    )

    rule = (
        alt.Chart()
        .mark_rule(color="black", clip=True)
        .encode(
            x=alt.datum(0),
            y=alt.datum(0),
            x2=alt.datum(100),
            y2=alt.datum(100),
        )
    )

    legend = (
        alt.Chart(zz)
        .mark_point(clip=True)
        .encode(alt.Y(f"{carrier}:N", title="Carrier").axis(orient="right"), color=color)
        .add_params(selection)
    )

    return ((scatter + rule).properties(width=600, height=600) | legend).interactive()
