import logging
from typing import Literal

import numpy as np
import pandas as pd

from .database import Database

logger = logging.getLogger("passengersim.database")


def fare_class_mix(
    cnx: Database, scenario: str, burn_samples: int = 100
) -> pd.DataFrame:
    """
    Fare class mix by carrier.

    This query requires that the simulation was run while recording final fare
    details (i.e. with the `fare` or `fare_final` flags set on `Config.db.write_items`).

    Parameters
    ----------
    cnx : Database
    scenario : str
    burn_samples : int, default 100
        The average total demand will be computed ignoring this many samples from the
        beginning of each trial.

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe is indexed by `carrier` and `booking_class`, and
        has these columns:

        - `avg_sold`: Average number of sales in this booking class.
        - `avg_revenue`: Average total revenue earned from customers booking in
            this booking class.
        - `avg_price`: Average price per ticket from customers booking in this
            booking class.
    """
    qry = """
    SELECT carrier, booking_class,
           (AVG(sold)) AS avg_sold,
           (AVG(revenue)) AS avg_revenue,
           (AVG(revenue) / AVG(sold)) AS avg_price
    FROM (
            SELECT
                trial, scenario, carrier, booking_class,
                SUM(sold) AS sold,
                SUM(sold * price) AS revenue
            FROM
                fare_detail LEFT JOIN fare_defs USING (fare_id)
            WHERE
                days_prior = 0
                AND sample >= ?2
                AND scenario = ?1
            GROUP BY
                trial, sample, carrier, booking_class
    ) tmp
    GROUP BY carrier, booking_class
    ORDER BY carrier, booking_class;
    """
    return cnx.dataframe(qry, (scenario, burn_samples)).set_index(
        ["carrier", "booking_class"]
    )


def od_fare_class_mix(
    cnx: Database, orig: str, dest: str, scenario: str, burn_samples: int = 100
) -> pd.DataFrame:
    """
    Fare class mix by carrier for a particular origin-destination market.

    This query requires that the simulation was run while recording final fare
    details (i.e. with the `fare` or `fare_final` flags set on `Config.db.write_items`).

    Parameters
    ----------
    cnx : Database
    orig, dest : str
        Origin and destination to query.
    scenario : str
    burn_samples : int, default 100
        The average total demand will be computed ignoring this many samples from the
        beginning of each trial.

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe is indexed by `carrier` and `booking_class`, and
        has these columns:

        - `avg_sold`: Average number of sales in this booking class.
        - `avg_revenue`: Average total revenue earned from customers booking in
            this booking class.
        - `avg_price`: Average price per ticket from customers booking in this
            booking class.
    """

    qry = """
    SELECT carrier, booking_class,
           (AVG(sold)) AS avg_sold,
           (AVG(revenue)) AS avg_revenue,
           (AVG(revenue) / AVG(sold)) AS avg_price
    FROM (
            SELECT
                trial, scenario, carrier, booking_class,
                SUM(sold) AS sold,
                SUM(sold * price) AS revenue,
                COUNT(*) AS nobs
            FROM
                fare_detail LEFT JOIN fare_defs USING (fare_id)
            WHERE
                days_prior = 0
                AND sample >= ?2
                AND scenario = ?1
                AND orig = ?3
                AND dest = ?4
            GROUP BY
                trial, sample, carrier, booking_class
    ) tmp
    GROUP BY carrier, booking_class
    ORDER BY carrier, booking_class;
    """
    return cnx.dataframe(qry, (scenario, burn_samples, orig, dest)).set_index(
        ["carrier", "booking_class"]
    )


def load_factors(cnx: Database, scenario: str, burn_samples: int = 100) -> pd.DataFrame:
    qry = """
    SELECT carrier,
           ROUND(AVG(sold)) AS avg_legs_sold,
           ROUND(AVG(100.0 * sold / cap), 2) AS avg_leg_lf,
           ROUND(AVG(100.0 * rpm / asm), 2) AS sys_lf,
           ROUND(AVG(revenue), 2) AS avg_rev,
           ROUND(AVG(revenue / asm), 3) AS yield,
           ROUND(AVG(revenue) / AVG(sold)) AS avg_leg_price,
           COUNT(*) AS n_obs
    FROM (SELECT trial, sample, carrier,
                 SUM(sold) AS sold,
                 SUM(capacity) AS cap,
                 SUM(sold * distance) AS rpm,
                 SUM(capacity * distance) AS asm,
                 SUM(revenue) AS revenue
          FROM leg_detail
                   JOIN leg_defs USING (flt_no)
          WHERE days_prior = 0
            AND sample >= ?2
            AND scenario = ?1
          GROUP BY trial, sample, carrier
         ) tmp
    GROUP BY carrier
    """
    return cnx.dataframe(qry, (scenario, burn_samples))


def load_factor_distribution(
    cnx: Database,
    scenario: str,
    burn_samples: int = 100,
    cutoffs=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95),
) -> pd.DataFrame:
    cutoffs = sorted([float(j) for j in cutoffs])
    if 0.0 not in cutoffs:
        cutoffs = [0.0] + cutoffs
    if 1.0 not in cutoffs:
        cutoffs = cutoffs + [1.0]
    vars = []
    for i in range(len(cutoffs) - 2):
        vars.append(
            f"count(CASE WHEN lf>= {cutoffs[i]} AND lf < {cutoffs[i+1]} THEN 1 END) "
            f"AS '{cutoffs[i]} - {cutoffs[i+1]}'"
        )
    vars.append(
        f"count(CASE WHEN lf>= {cutoffs[-2]} AND lf <= {cutoffs[-1]} THEN 1 END) "
        f"AS '{cutoffs[-2]} - {cutoffs[-1]}'"
    )
    vars = ",\n        ".join(vars)
    qry = f"""
    SELECT
        carrier,
        {vars}
    FROM (
        SELECT carrier, sold, capacity, (1.0*sold)/capacity AS lf
        FROM leg_detail
                   JOIN leg_defs USING (flt_no)
          WHERE days_prior = 0  -- only after all sales are recorded
            AND sample >= ?2    -- only after burn period
            AND scenario = ?1   -- only for this scenario
    )
    GROUP BY carrier
    """
    return cnx.dataframe(qry, (scenario, burn_samples))


def total_demand(cnx: Database, scenario: str, burn_samples: int = 100) -> float:
    """
    Average total demand.

    This query requires that the simulation was run while recording final demand
    details (i.e. with the `demand` or `demand_final` flags set on
    `Config.db.write_items`).

    Parameters
    ----------
    cnx : Database
    scenario : str
    burn_samples : int, default 100
        The average total demand will be computed ignoring this many samples from the
        beginning of each trial.

    Returns
    -------
    float
    """
    qry = """
    SELECT AVG(sample_demand)
    FROM (
        SELECT
            trial, sample, SUM(sample_demand) AS sample_demand
        FROM
            demand_detail
        WHERE
            days_prior = 0
            AND sample >= ?2
            AND scenario = ?1
        GROUP BY
            trial, sample) tmp;
    """
    return cnx.dataframe(qry, (scenario, burn_samples)).iloc[0, 0]


def bookings_by_timeframe(
    cnx: Database,
    scenario: str,
    from_fare_detail: bool = False,
    burn_samples: int = 100,
) -> pd.DataFrame:
    """
    Average bookings and revenue by carrier, booking class, and timeframe.

    This query requires that the simulation was run while recording supporting
    details (i.e. with the `bookings` or `fare` flags set on `Config.db.write_items`).

    Parameters
    ----------
    cnx : Database
    scenario : str
    from_fare_detail : bool, default False
        Reconstruct this table from the `fare_detail` table.  This is generally
        slower than accessing the `bookings` table directly, and also requires
        substantially more data to have been saved into the database by setting
        the `fare` flag on `Config.db.write_items`
    burn_samples : int, default 100
        The bookings will be computed ignoring this many samples from the
        beginning of each trial. This argument is nominally ignored by this query
        unless `from_fare_detail` is true, although the simulator will have already
        ignored the burned samples when storing the data in the bookings table.

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe is indexed by `trial`, `carrier`, `class`,
        and `days_prior`, and has these columns:

        - `avg_sold`: Average number of sales.
        - `avg_business`: Average number of sales to passengers in the business segment.
        - `avg_leisure`: Average number of sales to leisure passengers.
        - `avg_revenue`: Average total revenue earned from customers booking in this
            booking class in this time period.
        - `avg_price`: Average price per ticket from customers booking in this booking
            class in this time period
    """
    qry_fare = """
    SELECT trial, carrier, booking_class, days_prior,
           (AVG(sold)) AS avg_sold,
           (AVG(sold_business)) AS avg_business,
           (AVG(sold_leisure)) AS avg_leisure,
           (AVG(revenue)) AS avg_revenue,
           (AVG(revenue) / AVG(sold)) AS avg_price,
           (SUM(sold)) AS tot_sold
    FROM (SELECT trial, scenario, carrier, booking_class, days_prior,
                 SUM(sold) AS sold,
                 SUM(sold_business) AS sold_business,
                 SUM(sold - sold_business) AS sold_leisure,
                 SUM(sold * price) AS revenue
          FROM fare_detail LEFT JOIN fare_defs USING (fare_id)
          WHERE
                sample >= ?2
                AND scenario = ?1
          GROUP BY trial, sample, carrier, booking_class, days_prior) a
    GROUP BY carrier, booking_class, days_prior, trial
    ORDER BY carrier, booking_class, days_prior, trial;
    """

    if from_fare_detail:
        return cnx.dataframe(qry_fare, (scenario, burn_samples)).set_index(
            ["trial", "carrier", "booking_class", "days_prior"]
        )

    qry_bookings = """
    SELECT
        trial,
        carrier,
        booking_class,
        days_prior,
        avg_sold,
        avg_business,
        avg_leisure,
        avg_revenue,
        avg_price
    FROM
        bookings_by_timeframe
    WHERE
        scenario = ?1
    GROUP BY
        carrier, booking_class, days_prior, trial
    ORDER BY
        carrier, booking_class, days_prior, trial;
    """
    return cnx.dataframe(qry_bookings, (scenario,)).set_index(
        ["trial", "carrier", "booking_class", "days_prior"]
    )


def leg_forecasts(
    cnx: Database, scenario: str, burn_samples: int = 100
) -> pd.DataFrame:
    """
    Average forecasts of demand by leg, bucket, and days to departure.

    This query requires that the simulation was run while recording leg bucket
    details (i.e. with the `bucket` flag set on `Config.db.write_items`).

    Parameters
    ----------
    cnx : Database
    scenario : str
    burn_samples : int, default 100
        The forecasts will be analyzed ignoring this many samples from the
        beginning of each trial.

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe is indexed by `carrier`, `flt_no`,
        `bucket_number`, `booking_class` and `days_prior`, and has these columns:

        - `forecast_mean`: Average forecast mean (mu).
        - `forecast_stdev`: Average forecast standard deviation (sigma).
        - `forecast_closed_in_tf`: Average fraction of time the timeframe was
            closed in the data used to make a forecast.
        - `forecast_closed_in_tf`: Average fraction of time any future timeframe
            was closed in the data used to make a forecast.
    """
    qry = """
    SELECT
        carrier,
        flt_no,
        bucket_number,
        name as booking_class,
        days_prior,
        AVG(forecast_mean) as forecast_mean,
        AVG(forecast_stdev) as forecast_stdev,
        AVG(forecast_closed_in_tf) as forecast_closed_in_tf,
        AVG(forecast_closed_in_future) as forecast_closed_in_future
    FROM
        leg_bucket_detail LEFT JOIN leg_defs USING (flt_no)
    WHERE
        scenario = ?1
        AND sample >= ?2
    GROUP BY
        carrier, flt_no, bucket_number, name, days_prior
    """
    return cnx.dataframe(
        qry,
        (
            scenario,
            burn_samples,
        ),
    ).set_index(["carrier", "flt_no", "bucket_number", "booking_class", "days_prior"])


def path_forecasts(
    cnx: Database, scenario: str, burn_samples: int = 100
) -> pd.DataFrame:
    """
    Average forecasts of demand by path, class, and days to departure.

    This query requires that the simulation was run while recording path-class
    details (i.e. with the `pathclass` flag set on `Config.db.write_items`).

    Parameters
    ----------
    cnx : Database
    scenario : str
    burn_samples : int, default 100
        The forecasts will be analyzed ignoring this many samples from the
        beginning of each trial.

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe is indexed by `path_id`, `booking_class` and
        `days_prior`, and has these columns:

        - `forecast_mean`: Average forecast mean (mu).
        - `forecast_stdev`: Average forecast standard deviation (sigma).
        - `forecast_closed_in_tf`: Average fraction of time the timeframe was
            closed in the data used to make a forecast.
        - `forecast_closed_in_tf`: Average fraction of time any future timeframe
            was closed in the data used to make a forecast.
    """
    qry = """
    SELECT
        path_id,
        booking_class,
        days_prior,
        AVG(forecast_mean) as forecast_mean,
        AVG(forecast_stdev) as forecast_stdev,
        AVG(forecast_closed_in_tf) as forecast_closed_in_tf,
        AVG(forecast_closed_in_future) as forecast_closed_in_future
    FROM
        path_class_detail
    WHERE
        scenario = ?1
        AND sample >= ?2
    GROUP BY
        path_id, booking_class, days_prior
    """
    return cnx.dataframe(
        qry,
        (
            scenario,
            burn_samples,
        ),
    ).set_index(["path_id", "booking_class", "days_prior"])


def demand_to_come(
    cnx: Database, scenario: str, burn_samples: int = 100
) -> pd.DataFrame:
    """
    Demand by market and timeframe across each sample.

    This query delivers sample-by-sample timeframe demand results for the
    various markets (origin, destination, passenger type) in the simulation.
    It requires that the simulation was run while recording demand details
    (i.e. with the `demand` flag set on `Config.db.write_items`).

    Parameters
    ----------
    cnx : Database
    scenario : str
    burn_samples : int, default 100
        The demand will be returned ignoring this many samples from the
        beginning of each trial.

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe is indexed by `iteration`, `trial`, `sample`,
        `segment`, `orig`, and `dest`; and has columns defined by the DCPs.
        The values stored are the total remaining demand to come at each DCP.
    """
    # Provides content similar to PODS *.DHS output file, but with market level detail
    qry = """
    SELECT
        iteration, trial, sample, segment, orig, dest, days_prior, sold, no_go,
        (round(sample_demand) - sold - no_go) AS future_demand
    FROM
        demand_detail
    WHERE
        scenario = ?1
        AND sample >= ?2
    """
    dmd = cnx.dataframe(
        qry, (scenario, burn_samples), dtype={"future_demand": np.int32}
    )
    dhs = (
        dmd.set_index(
            ["iteration", "trial", "sample", "segment", "orig", "dest", "days_prior"]
        )["future_demand"]
        .unstack("days_prior")
        .sort_values(by="days_prior", axis=1, ascending=False)
    )
    return dhs


def carrier_history(
    cnx: Database, scenario: str, burn_samples: int = 100
) -> pd.DataFrame:
    """
    Sample-level details of carrier-level measures.

    This query delivers sample-by-sample aggregated summary results for the
    various carriers in the simulation. It requires that the simulation was
    run while recording leg bucket details (i.e. with the `bucket` flag set
    on `Config.db.write_items`).

    Parameters
    ----------
    cnx : Database
    scenario : str
    burn_samples : int, default 100
        The history will be returned ignoring this many samples from the
        beginning of each trial.

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe is indexed by `iteration`, `trial` and `sample`,
        and columns defined with a two-level MultiIndex.  The second level of
        the columns MultiIndex represents the carriers, while the top level
        includes these columns:

        - `forecast_mean`: Forecast mean (mu) at the beginning of the booking
            curve, summed over all this carrier's legs in this sample.
        - `forecast_stdev`: Forecast standard deviation (sigma) at the beginning
            of the booking curve, aggregated over all this carrier's legs in this
            sample.
        - `sold`: Total bookings accepted by this carrier in this sample.
        - `revenue`: Total revenue for this carrier in this sample.
    """
    # Provides content similar to PODS *.HST output file.
    max_rrd = int(
        cnx.dataframe(
            """
            SELECT max(days_prior) FROM leg_bucket_detail WHERE scenario == ?1
            """,
            (scenario,),
        ).iloc[0, 0]
    )
    bd1 = cnx.dataframe(
        """
        SELECT
            iteration, trial, sample, carrier,
            sum(forecast_mean) as forecast_mean,
            sqrt(sum(forecast_stdev*forecast_stdev)) as forecast_stdev
        FROM leg_bucket_detail LEFT JOIN leg_defs USING (flt_no)
        WHERE days_prior == ?2 AND scenario == ?1 AND sample >= ?3
        GROUP BY iteration, trial, sample, carrier
        """,
        (scenario, max_rrd, burn_samples),
    ).set_index(["iteration", "trial", "sample", "carrier"])
    bd2 = cnx.dataframe(
        """
        SELECT
            iteration, trial, sample, carrier,
            sum(sold) as sold,
            sum(revenue) as revenue
        FROM leg_bucket_detail LEFT JOIN leg_defs USING (flt_no)
        WHERE days_prior == 0 AND scenario == ?1 AND sample >= ?2
        GROUP BY iteration, trial, sample, carrier
        """,
        (scenario, burn_samples),
    ).set_index(["iteration", "trial", "sample", "carrier"])
    return pd.concat([bd1, bd2], axis=1).unstack("carrier")


def bid_price_history(
    cnx: Database,
    scenario: str,
    burn_samples: int = 100,
    weighting: Literal["equal", "capacity"] = "equal",
) -> pd.DataFrame:
    """
    Compute average bid price history over all legs for each carrier.

    This query requires that the simulation was run while recording leg
    details (i.e. with the `leg` flag set on `Config.db.write_items`),
    including bid prices.

    Parameters
    ----------
    cnx : Database
    scenario : str
    burn_samples : int, default 100
        The bid prices will be analyzed ignoring this many samples from the
        beginning of each trial.
    weighting : {'equal', 'capacity'}, default 'equal'
        How to weight the bid prices.  If 'equal', then each leg is weighted
        equally.  If 'capacity', then each leg is weighted by its total capacity.

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe is indexed by `carrier` and `days_prior`, and has
        these columns:

        - `bid_price_mean`: Average bid price across all samples and all legs
        - `bid_price_stdev`: Sample standard deviation of bid prices across all
            samples and all legs
        - `fraction_some_cap`: Fraction of all legs across all samples that have
            non-zero capacity available for sale.
        - `fraction_zero_cap`: Fraction of all legs across all samples that have
            zero capacity available for sale.  Bid prices are computed for these
            legs but are not really meaningful.
        - `some_cap_bid_price_mean`: Average bid price across all samples and
            all legs conditional on the leg having non-zero capacity.
        - `some_cap_bid_price_stdev`: Sample standard deviation of bid prices
            across all samples and all legs conditional on the leg having
            non-zero capacity.

    """
    if weighting not in ("equal", "capacity"):
        raise ValueError(f"unknown weighting {weighting}")
    qry = """
    SELECT
        carrier,
        days_prior,
        avg(bid_price) as bid_price_mean,
        stdev(bid_price) as bid_price_stdev,
        avg(CASE WHEN leg_detail.sold < leg_defs.capacity THEN 1.0 ELSE 0.0 END)
            as fraction_some_cap,
        avg(CASE WHEN leg_detail.sold < leg_defs.capacity THEN 0.0 ELSE 1.0 END)
            as fraction_zero_cap
    FROM leg_detail
        LEFT JOIN leg_defs ON leg_detail.flt_no = leg_defs.flt_no
    WHERE
        scenario == ?1
        AND sample >= ?2
    GROUP BY
        carrier, days_prior
    """
    bph = cnx.dataframe(
        qry,
        (
            scenario,
            burn_samples,
        ),
    )
    qry2 = """
    SELECT
        carrier,
        days_prior,
        avg(bid_price) as some_cap_bid_price_mean_unweighted,
        stdev(bid_price) as some_cap_bid_price_stdev,
        (SUM(bid_price * leg_defs.capacity) / SUM(leg_defs.capacity))
            as some_cap_bid_price_mean_capweighted
    FROM leg_detail
        LEFT JOIN leg_defs ON leg_detail.flt_no = leg_defs.flt_no
    WHERE
        scenario == ?1
        AND sample >= ?2
        AND leg_detail.sold < leg_defs.capacity
    GROUP BY
        carrier, days_prior
    """
    bph_some_cap = cnx.dataframe(
        qry2,
        (
            scenario,
            burn_samples,
        ),
    ).set_index(["carrier", "days_prior"])
    bph = bph.set_index(["carrier", "days_prior"]).join(bph_some_cap)
    bph = bph.sort_index(ascending=(True, False))
    if weighting == "equal":
        bph["some_cap_bid_price_mean"] = bph["some_cap_bid_price_mean_unweighted"]
    elif weighting == "capacity":
        bph["some_cap_bid_price_mean"] = bph["some_cap_bid_price_mean_capweighted"]
    else:
        raise ValueError(f"unknown weighting {weighting}")
    return bph


def displacement_history(
    cnx: Database,
    scenario: str,
    burn_samples: int = 100,
) -> pd.DataFrame:
    """
    Compute average displacement cost history over all legs for each carrier.

    This query requires that the simulation was run while recording leg
    details (i.e. with the `leg` flag set on `Config.db.write_items`),
    including displacement costs.

    Parameters
    ----------
    cnx : Database
    scenario : str
    burn_samples : int, default 100
        The bid prices will be analyzed ignoring this many samples from the
        beginning of each trial.

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe is indexed by `carrier` and `days_prior`, and has
        these columns:

        - `displacement_mean`: Average displacement cost across all samples and
            all legs
        - `displacement_stdev`: Sample standard deviation of displacement cost
            across all samples and all legs
    """
    qry = """
    SELECT
        carrier,
        days_prior,
        avg(displacement) as displacement_mean,
        stdev(displacement) as displacement_stdev
    FROM leg_detail
        LEFT JOIN leg_defs ON leg_detail.flt_no = leg_defs.flt_no
    WHERE
        scenario == ?1
        AND sample >= ?2
    GROUP BY
        carrier, days_prior
    ORDER BY
        carrier, days_prior DESC
    """
    df = cnx.dataframe(
        qry,
        (
            scenario,
            burn_samples,
        ),
    )
    df = df.set_index(["carrier", "days_prior"])
    return df


def local_and_flow_yields(
    cnx: Database, scenario: str, burn_samples: int = 100
) -> pd.DataFrame:
    """
    Compute yields for local (nonstop) and flow (connecting) passengers.

    This query requires that the simulation was run while recording path class
    details (i.e. with the `pathclass` or `pathclass_final` flags set on
    `Config.db.write_items`).

    Parameters
    ----------
    cnx : Database
    scenario : str
    burn_samples : int, default 100
        The yields will be computed ignoring this many samples from the
        beginning of each trial.

    Returns
    -------
    pandas.DataFrame
    """
    qry = """
    WITH path_yields AS (
        SELECT
            iteration, trial, sample, path_id, leg1, leg2,
            SUM(sold) as total_sold,
            SUM(revenue) as total_revenue,
            distance,
            SUM(revenue) / (SUM(sold) * distance) AS yield,
            leg2 IS NULL AS local
        FROM
            path_class_detail
            LEFT JOIN path_defs USING (path_id)
        WHERE
            days_prior == 0
            AND scenario == ?1
            AND sample >= ?2
        GROUP BY
            path_id
    )
    SELECT
        flt_no, carrier, orig, dest, capacity, leg_defs.distance,
        yield AS local_yield,
        CAST(total_sold AS REAL) /
            (total_sold + IFNULL(f1.flow_sold, 0) + IFNULL(f2.flow_sold, 0))
            AS local_fraction,
        (IFNULL(f1.flow_revenue, 0) + IFNULL(f2.flow_revenue, 0))
            / (IFNULL(f1.flow_rpm, 0) + IFNULL(f2.flow_rpm, 0))
            AS flow_yield
    FROM
        leg_defs
        LEFT JOIN path_yields locals
        ON locals.leg1 == flt_no AND locals.leg2 IS NULL
        LEFT JOIN (
            SELECT
                leg1,
                SUM(total_sold) AS flow_sold,
                SUM(total_revenue) AS flow_revenue,
                SUM(total_sold * distance) AS flow_rpm
            FROM
                path_yields
            WHERE
                leg2 IS NOT NULL
            GROUP BY leg1
        ) f1 ON f1.leg1 == leg_defs.flt_no
        LEFT JOIN (
            SELECT
                leg2,
                SUM(total_sold) AS flow_sold,
                SUM(total_revenue) AS flow_revenue,
                SUM(total_sold * distance) AS flow_rpm
            FROM
                path_yields
            GROUP BY leg2
        ) f2 ON f2.leg2 == leg_defs.flt_no
    """
    df = cnx.dataframe(
        qry,
        (
            scenario,
            burn_samples,
        ),
    )
    return df


def leg_local_and_flow_by_class(
    cnx: Database, scenario: str, burn_samples: int = 100
) -> pd.DataFrame:
    logger.info("creating pthcls temp table")
    cnx.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS pthcls AS
        SELECT
            sold, leg1, booking_class, iteration, trial, sample
        FROM
            path_class_detail LEFT JOIN path_defs USING(path_id)
        WHERE
            days_prior == 0
            AND leg2 IS NULL
            AND scenario == ?1
            AND sample >= ?2
        """,
        (
            scenario,
            burn_samples,
        ),
    )

    logger.info("indexing pthcls temp table")
    cnx.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pthcls_1 ON pthcls (
            iteration, trial, sample, leg1, booking_class
        );
        """
    )

    logger.info("running leg_local_and_flow_by_class query")
    qry = """
    SELECT
        flt_no,
        leg_defs.carrier,
        leg_defs.orig,
        leg_defs.dest,
        leg_bucket_detail.name as booking_class,
        AVG(leg_bucket_detail.sold) AS carried_all,
        IFNULL(AVG(pthcls.sold), 0) AS carried_loc
    FROM
        leg_bucket_detail
        LEFT JOIN leg_defs USING (flt_no)
        LEFT JOIN pthcls ON
            flt_no == pthcls.leg1
            AND leg_bucket_detail.name == pthcls.booking_class
            AND leg_bucket_detail.iteration == pthcls.iteration
            AND leg_bucket_detail.trial == pthcls.trial
            AND leg_bucket_detail.sample == pthcls.sample
    WHERE
        leg_bucket_detail.scenario == ?1
        AND leg_bucket_detail.sample >= ?2
        AND days_prior == 0
    GROUP BY
        flt_no,
        leg_defs.carrier,
        leg_defs.orig,
        leg_defs.dest,
        leg_bucket_detail.name
    """
    df = cnx.dataframe(
        qry,
        (
            scenario,
            burn_samples,
        ),
    )
    return df
