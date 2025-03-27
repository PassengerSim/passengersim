import altair as alt
import pandas as pd

from passengersim.driver import Simulation
from passengersim.reporting import write_trace


def fig_bid_prices_by_leg(
    sim: Simulation,
    *,
    trace: pd.ExcelWriter | None = None,
):
    """
    Plot average bid prices by leg and days prior to departure.

    This plot will get messy for large numbers of flights, but it's useful for
    understanding the distribution of bid prices by days prior to departure on
    small networks.

    Parameters
    ----------
    sim
    trace

    Returns
    -------
    alt.Chart
    """
    burn = sim.sim.config.simulation_controls.burn_samples

    df = sim.cnx.dataframe(
        """
        SELECT
          flt_no,
          days_prior,
          AVG(bid_price) as bid_price
        FROM leg_detail
        WHERE sample >= ?1
        GROUP BY
          flt_no,
          days_prior
        """,
        (burn,),
    )

    if trace is not None:
        write_trace(trace, df, basesheetname="forecasts")

    line_encoding = dict(
        x=alt.X("days_prior:Q").scale(reverse=True).title("Days Prior to Departure"),
        y=alt.Y("bid_price", title="Bid Price"),
        color=alt.Color("flt_no:N"),
    )
    chart = alt.Chart(df)
    fig = chart.mark_line(interpolate="step-before").encode(**line_encoding)

    return fig
