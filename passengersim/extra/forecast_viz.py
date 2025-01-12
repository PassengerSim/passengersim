import altair as alt
import pandas as pd

from passengersim.driver import Simulation
from passengersim.reporting import write_trace


def fig_forecasts_and_bid_prices(
    sim: Simulation,
    trial: int = 0,
    days_prior: int = 63,
    leg_id: int = 101,
    joint_scale: bool = False,
    *,
    trace: pd.ExcelWriter | None = None,
):
    burn = sim.sim.config.simulation_controls.burn_samples

    bp_df = sim.cnx.dataframe(
        """
        SELECT
          sample,
          name as booking_class,
          forecast_mean
        FROM leg_bucket_detail
        WHERE sample >= ?1
          AND trial == ?2
          AND days_prior == ?3
          AND leg_id == ?4
        """,
        (burn, trial, days_prior, leg_id),
    )

    lg_df = sim.cnx.dataframe(
        """
        SELECT
          sample,
          bid_price
        FROM leg_detail
        WHERE sample >= ?1
          AND trial == ?2
          AND days_prior == ?3
          AND leg_id == ?4
        """,
        (burn, trial, days_prior, leg_id),
    )

    lg_df = lg_df.join(bp_df.groupby("sample").forecast_mean.sum(), on="sample")

    chart = alt.Chart(bp_df, height=40)
    if trace is not None:
        write_trace(trace, bp_df, basesheetname="forecasts")
    forecast = chart.mark_area().encode(
        x="sample:Q",
        y=alt.Y(
            "forecast_mean:Q",
            scale=alt.Scale(zero=joint_scale),
            axis=alt.Axis(title=None),
        ),
        color="booking_class:N",
        row=alt.Row("booking_class:N", spacing=0)
        .title(None)
        .header(labelAngle=0, labelAlign="left", labelPadding=5, labelFont="bold"),
    )
    if not joint_scale:
        forecast = forecast.resolve_scale(y="independent")

    chart2 = alt.Chart(lg_df, height=80)
    if trace is not None:
        write_trace(trace, lg_df, basesheetname="bid_prices")
    bidprice = chart2.mark_line(color="#00cf37").encode(
        x="sample:Q",
        y=alt.Y(
            "bid_price:Q",
            scale=alt.Scale(zero=False),
            axis=alt.Axis(title="Bid Price", titleColor="#00cf37"),
        ),
    )
    agg_forecast = chart2.mark_line(color="#5e5e5e").encode(
        x="sample:Q",
        y=alt.Y(
            "forecast_mean:Q",
            scale=alt.Scale(zero=False),
            axis=alt.Axis(title="Agg. Forecast", titleColor="#5e5e5e"),
        ),
    )

    return forecast & (bidprice + agg_forecast).resolve_scale(y="independent")
