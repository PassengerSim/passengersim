from __future__ import annotations

from collections.abc import Iterable

from .database import Database


def create_table_configs(cnx: Database):
    sql = """
    CREATE TABLE IF NOT EXISTS runtime_configs (
        scenario		TEXT PRIMARY KEY,
        pxsim_version   TEXT,
        configs         TEXT,
        updated_at	    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """
    cnx.execute(sql)


def create_table_leg_defs(cnx: Database, legs: Iterable | None = None):
    sql = """
    CREATE TABLE IF NOT EXISTS leg_defs
    (
        flt_no INTEGER PRIMARY KEY,
        carrier TEXT,
        orig TEXT,
        dest TEXT,
        dep_time INTEGER,
        arr_time INTEGER,
        capacity INTEGER,
        distance FLOAT
    );
    """
    cnx.execute(sql)
    for leg in legs:
        cnx.execute(
            """
            INSERT OR REPLACE INTO leg_defs(
                flt_no,
                carrier,
                orig,
                dest,
                dep_time,
                arr_time,
                capacity,
                distance
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8
            )
            """,
            (
                leg.flt_no,
                leg.carrier,
                leg.orig,
                leg.dest,
                leg.dep_time,
                leg.arr_time,
                leg.capacity,
                leg.distance,
            ),
        )


def create_table_fare_defs(cnx: Database, fares: Iterable | None = None) -> None:
    """
    Create and populate a static database table of fares.

    The contents of this table is static input data, used but not mutated
    by the simulator.

    Parameters
    ----------
    cnx : Database
    fares : Iterable[Fare]
        The fares to store in the database.
    """
    sql = """
    CREATE TABLE IF NOT EXISTS fare_defs
    (
        fare_id       INTEGER PRIMARY KEY,
        carrier       VARCHAR(10) NOT NULL,
        orig          VARCHAR(10) NOT NULL,
        dest          VARCHAR(10) NOT NULL,
        booking_class VARCHAR(10) NOT NULL,
        price         FLOAT NOT NULL,
        restrictions  VARCHAR(20) NOT NULL,
        category      VARCHAR(20)
    );
    """
    cnx.execute(sql)
    sql2 = """
    CREATE INDEX IF NOT EXISTS fare_defs_idx_2
    ON fare_defs (
        carrier, booking_class
    );
    """
    cnx.execute(sql2)
    for fare in fares:
        cnx.execute(
            """
            INSERT OR REPLACE INTO fare_defs(
                fare_id,
                carrier,
                orig,
                dest,
                booking_class,
                price,
                restrictions,
                category
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8
            )
            """,
            (
                fare.fare_id,
                fare.carrier,
                fare.orig,
                fare.dest,
                fare.booking_class,
                fare.price,
                ",".join(
                    [
                        fare.get_restriction(pos)
                        for pos in range(fare.num_restrictions())
                    ]
                ),
                fare.category,
            ),
        )


def create_table_path_defs(cnx: Database, paths: Iterable | None = None):
    sql = """
    CREATE TABLE IF NOT EXISTS path_defs
    (
        path_id INTEGER PRIMARY KEY,
        carrier TEXT,
        orig TEXT,
        stop1 TEXT,
        dest TEXT,
        leg1 INTEGER,
        leg2 INTEGER,
        distance FLOAT
    );
    """
    cnx.execute(sql)
    for pth in paths:
        connects = pth.num_legs() > 1
        cnx.execute(
            """
            INSERT OR REPLACE INTO path_defs(
                path_id,
                carrier,
                orig,
                stop1,
                dest,
                leg1,
                leg2,
                distance
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8
            )
            """,
            (
                pth.path_id,
                pth.get_leg_carrier(0),
                pth.orig,
                pth.get_leg_dest(0) if connects else None,
                pth.dest,
                pth.get_leg_fltno(0),
                pth.get_leg_fltno(1) if connects else None,
                pth.get_total_distance(),
            ),
        )


def create_table_leg_detail(cnx: Database, primary_key: bool = False) -> None:
    """
    Create the `leg_detail` table in the database.

    Parameters
    ----------
    cnx : Database
    primary_key : bool, default False
    """
    sql = """
    CREATE TABLE IF NOT EXISTS leg_detail
    (
        scenario	    	VARCHAR(20) NOT NULL,
        iteration	    	INT NOT NULL,
        trial	        	INT NOT NULL,
        sample  	    	INT NOT NULL,
        days_prior       	INT NOT NULL,
        flt_no		    	INT NOT NULL,
        updated_at	    	DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        sold	    		INT,
        revenue             FLOAT,
        q_demand            FLOAT,
        untruncated_demand  FLOAT,
        forecast_mean       FLOAT,
        bid_price           FLOAT,
        displacement        FLOAT
        {primary_key}
    );
    """
    if primary_key is True:
        sql = sql.format(
            primary_key=", PRIMARY KEY(scenario, iteration, trial, sample, "
            "carrier, orig, dest, flt_no, days_prior, dep_date)"
        )
    else:
        sql = sql.format(primary_key="")
    cnx.execute(sql)


def create_table_leg_bucket_detail(cnx: Database, primary_key: bool = False):
    sql = """
    CREATE TABLE IF NOT EXISTS leg_bucket_detail
    (
        scenario		VARCHAR(20) NOT NULL,
        iteration		INT NOT NULL,
        trial	    	INT NOT NULL,
        sample  		INT NOT NULL,
        days_prior   	INT NOT NULL,
        flt_no			INT NOT NULL,
        bucket_number   INT NOT NULL,
        name            VARCHAR(10) NOT NULL,
        auth    		INT,
        revenue    		FLOAT,
        sold			INT,
        untruncated_demand     FLOAT,
        forecast_mean   FLOAT,
        forecast_stdev  FLOAT,
        forecast_closed_in_tf FLOAT,
        forecast_closed_in_future FLOAT,
        updated_at		DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        {primary_key}
    );
    """
    if primary_key is True:
        sql = sql.format(
            primary_key=", PRIMARY KEY(scenario, iteration, trial, sample, days_prior, "
            "carrier, orig, dest, flt_no, dep_date, bucket_number)"
        )
    else:
        sql = sql.format(primary_key="")
    cnx.execute(sql)


def create_table_demand_detail(cnx: Database, primary_key: bool = False):
    sql = """
    CREATE TABLE IF NOT EXISTS demand_detail
    (
        scenario		VARCHAR(20) NOT NULL,
        iteration		INT NOT NULL,
        trial	    	INT NOT NULL,
        sample  		INT NOT NULL,
        days_prior	    INT NOT NULL,
        segment			VARCHAR(10) NOT NULL,
        orig			VARCHAR(10) NOT NULL,
        dest			VARCHAR(10) NOT NULL,
        updated_at		DATETIME NOT NULL DEFAuLT CURRENT_TIMESTAMP,
        sample_demand   FLOAT,
        sold			INT,
        no_go			INT,
        revenue			FLOAT
        {primary_key}
    );
    """
    if primary_key is True:
        sql = sql.format(
            primary_key=", PRIMARY KEY(scenario, iteration, trial, "
            "sample, days_prior, segment, orig, dest)"
        )
    else:
        sql = sql.format(primary_key="")
    cnx.execute(sql)


def create_table_fare_detail(cnx: Database, primary_key: bool = False):
    sql = """
    CREATE TABLE IF NOT EXISTS fare_detail
    (
        scenario		VARCHAR(20) NOT NULL,
        iteration		INT NOT NULL,
        trial	    	INT NOT NULL,
        sample  		INT NOT NULL,
        days_prior 		INT NOT NULL,
        fare_id         INT NOT NULL,
        sold			INT,
        sold_business	INT,
        updated_at		DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        {primary_key}
    );
    """
    if primary_key is True:
        sql = sql.format(
            primary_key=", PRIMARY KEY(scenario, iteration, trial, sample, "
            "days_prior, carrier, orig, dest, booking_class)"
        )
    else:
        sql = sql.format(primary_key="")
    cnx.execute(sql)


def create_table_bookings_by_timeframe(cnx: Database, primary_key: bool = False):
    sql = """
    CREATE TABLE IF NOT EXISTS bookings_by_timeframe
    (
        scenario		VARCHAR(20) NOT NULL,
        trial       	INT NOT NULL,
        carrier			VARCHAR(10) NOT NULL,
        booking_class   VARCHAR(10) NOT NULL,
        days_prior		INT NOT NULL,
        tot_sold		FLOAT,
        avg_sold		FLOAT,
        avg_business	FLOAT,
        avg_leisure     FLOAT,
        avg_revenue     FLOAT,
        avg_price       FLOAT,
        updated_at		DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        {primary_key}
    );
    """
    if primary_key is True:
        sql = sql.format(
            primary_key=", PRIMARY KEY(scenario, carrier, booking_class, days_prior)"
        )
    else:
        sql = sql.format(primary_key="")
    cnx.execute(sql)


def create_table_booking_curve(cnx: Database, primary_key: bool = True):
    sql = """
    CREATE TABLE IF NOT EXISTS booking_curve (
        scenario		VARCHAR(20) NOT NULL,
        carrier			VARCHAR(10) NOT NULL,
        orig			VARCHAR(10) NOT NULL,
        dest			VARCHAR(10) NOT NULL,
        flt_no			INT NOT NULL,
        days_prior  	INT NOT NULL,
        ratio           FLOAT NOT NULL
        {primary_key}
    );
    """
    if primary_key is True:
        sql = sql.format(
            primary_key=", PRIMARY KEY(scenario, carrier, "
            "orig, dest, flt_no, days_prior)"
        )
    else:
        sql = sql.format(primary_key="")
    cnx.execute(sql)


def create_table_path_class_detail(cnx: Database, primary_key: bool = False):
    sql = """
    CREATE TABLE IF NOT EXISTS path_class_detail
    (
        scenario		VARCHAR(20) NOT NULL,
        iteration		INT NOT NULL,
        trial	    	INT NOT NULL,
        sample  		INT NOT NULL,
        days_prior   	INT NOT NULL,
        path_id			INT NOT NULL,
        booking_class   VARCHAR(10) NOT NULL,
        sold			INT,
        revenue         FLOAT,
        forecast_mean   FLOAT,
        forecast_stdev  FLOAT,
        forecast_closed_in_tf FLOAT,
        forecast_closed_in_future FLOAT,
        updated_at		DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        {primary_key}
    );
    """
    if primary_key is True:
        sql = sql.format(
            primary_key=", PRIMARY KEY(scenario, iteration, trial, sample, days_prior, "
            "path_id, booking_class)"
        )
    else:
        sql = sql.format(primary_key="")
    cnx.execute(sql)


def create_table_distance(cnx: Database, primary_key: bool = True):
    sql = """
    CREATE TABLE IF NOT EXISTS distance (
        orig			VARCHAR(10) NOT NULL,
        dest			VARCHAR(10) NOT NULL,
        miles           FLOAT
        {primary_key}
    );
    """
    if primary_key is True:
        sql = sql.format(primary_key=", PRIMARY KEY(orig, dest)")
    else:
        sql = sql.format(primary_key="")
    cnx.execute(sql)


def create_tables(cnx: Database, primary_keys: dict[str, bool] | None = None):
    pk = dict(
        leg=False,
        leg_bucket=False,
        demand=False,
        fare=False,
        booking_curve=True,
        distance=True,
        bookings=False,
        path_class=False,
    )
    if primary_keys is not None:
        pk.update(primary_keys)
    create_table_configs(cnx)
    create_table_leg_detail(cnx, pk["leg"])
    create_table_leg_bucket_detail(cnx, pk["leg_bucket"])
    create_table_demand_detail(cnx, pk["demand"])
    create_table_fare_detail(cnx, pk["fare"])
    create_table_bookings_by_timeframe(cnx, pk["bookings"])
    create_table_booking_curve(cnx, pk["booking_curve"])
    create_table_path_class_detail(cnx, pk["path_class"])
    create_table_distance(cnx, pk["distance"])
    cnx._commit_raw()
