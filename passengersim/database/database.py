#
# Utilities for reading and writing PassengerSim data
# Uses SQLITE
#
from __future__ import annotations

import logging
import math
import sqlite3
import string
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from passengersim.config import Config
from passengersim.core import SimulationEngine

logger = logging.getLogger("passengersim.database")


class _VarianceFunc:
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 1

    def step(self, value):
        if value is None:
            return
        tM = self.M
        self.M += (value - tM) / self.k
        self.S += (value - tM) * (value - self.M)
        self.k += 1

    def finalize(self):
        if self.k < 3:
            return None
        return self.S / (self.k - 2)


class _StdevFunc(_VarianceFunc):
    def finalize(self):
        if self.k < 3:
            return None
        return math.sqrt(self.S / (self.k - 2))


class Database:
    """A wrapper to manage transactions for PassengerSim on SQLite."""

    def __init__(
        self,
        engine: Literal["sqlite", None] = "sqlite",
        filename=None,
        pragmas: Iterable[str] = (),
        commit_count_delay: int | None = 250,
    ):
        self._connection = None
        self.engine = engine
        self.filename = filename
        self.pragmas = pragmas
        self._counter = 0
        self._commit_count_delay = commit_count_delay
        if self._commit_count_delay is not None:
            self.commit = self._commit_by_count
        else:
            self.commit = self._commit_raw
        self.open()

    def __getattr__(self, item):
        return getattr(self._connection, item)

    def open(self, filename: str | None = None):
        """Open the connection if it is not already open."""
        if self._connection is not None:
            raise ConnectionError("the connection is already open")
        self.filename = filename or self.filename
        if self.engine is None:
            self._connection = None
        elif self.engine == "sqlite" and self.filename is None:
            self._connection = None
        elif self.engine == "sqlite":
            if self.filename != ":memory:":
                Path(self.filename).parent.mkdir(exist_ok=True, parents=True)
            logger.info(f"connecting to sqlite database: {self.filename}")
            self._connection = sqlite3.Connection(self.filename)
            self._connection.create_aggregate("VARIANCE", 1, _VarianceFunc)
            self._connection.create_aggregate("STDEV", 1, _StdevFunc)
            for pragma in self.pragmas:
                self._connection.execute(f"PRAGMA {pragma};")
            self._connection.execute("BEGIN TRANSACTION;")
            logger.debug("initializing sqlite tables")
            from .tables import create_tables

            create_tables(self)
        else:
            raise NotImplementedError(f"unknown engine {self.engine!r}")

    def close(self):
        """Flush pending operations and close the connection."""
        if self._connection:
            if self._connection.in_transaction:
                self._connection.execute("COMMIT;")
            self._connection.close()
            self._connection = None

    def _commit_by_count(self):
        self._counter += 1
        if self._counter >= self._commit_count_delay:
            self._commit_raw()
            self._counter = 0

    def _commit_raw(self):
        if self._connection:
            if self._connection.in_transaction:
                self._connection.execute("COMMIT;")
            self._connection.execute("BEGIN TRANSACTION;")

    def __enter__(self):
        if self._connection:
            return self._connection.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._connection:
            return self._connection.__exit__(exc_type, exc_val, exc_tb)

    @property
    def is_open(self) -> bool:
        return self._connection is not None

    def sql_placeholders(self, n: int):
        """A parenthesis enclosed set of `n` placeholders for the selected engine."""
        if self.engine == "sqlite":
            x = "?"
        else:
            x = "%s"
        return "(" + ", ".join(x for _ in range(n)) + ")"

    def delete_experiment(self, name: str):
        if self.is_open:
            logger.debug(f"deleting existing scenario {name!r} from database")
            self.execute("DELETE FROM leg_detail WHERE scenario = ?", (name,))
            self.execute("DELETE FROM leg_bucket_detail WHERE scenario = ?", (name,))
            self.execute("DELETE FROM demand_detail WHERE scenario = ?", (name,))
            self.execute("DELETE FROM fare_detail WHERE scenario = ?", (name,))
            self._commit_raw()
        else:
            logger.debug(f"database not open, cannot delete {name!r}")

    def save_configs(self, cfg: Config) -> None:
        """Save configs into the database."""
        from passengersim import __version__

        self.execute(
            """
        INSERT OR REPLACE INTO runtime_configs(
            scenario, pxsim_version, configs
        ) VALUES (?1, ?2, ?3)
        """,
            (
                cfg.scenario,
                str(__version__),
                cfg.model_dump_json(
                    exclude={"db": "dcp_write_hooks", "raw_license_certificate": True}
                ),
            ),
        )

    def load_configs(self, scenario=None) -> Config:
        import json

        if scenario:
            rawjson = next(
                self.execute(
                    "SELECT configs, max(updated_at) FROM runtime_configs "
                    "WHERE scenario = ?1",
                    (scenario,),
                )
            )[0]
        else:
            rawjson = next(
                self.execute("SELECT configs, max(updated_at) FROM runtime_configs")
            )[0]

        from pydantic import ValidationError

        try:
            return Config.model_validate(json.loads(rawjson))
        except ValidationError as err:
            logger.error(f"error loading configs: {err}")
            logger.error("raw json: %s", json.loads(rawjson)["rm_systems"])
            raise

    def save_details(self: Database, sim: SimulationEngine, dcp: int):
        """
        Save details, can be done at each RRD/DCP and at the end of the run
        """
        if not sim.save_timeframe_details and dcp > 0:
            return
        if sim.config.db.fast and isinstance(self._connection, sqlite3.Connection):
            sim.write_to_sqlite(
                self._connection,
                dcp,
                store_bid_prices=sim.config.db.store_leg_bid_prices,
                store_displacements=sim.config.db.store_displacements,
            )
        else:
            for leg in sim.legs:
                if "leg" in sim.config.db.write_items:
                    save_leg(self, sim, leg, dcp)
                if "bucket" in sim.config.db.write_items:
                    save_leg_bucket_multi(self, sim, leg, dcp)
            if "fare" in sim.config.db.write_items:
                save_fare_multi(self, sim, dcp)
            if "demand" in sim.config.db.write_items:
                save_demand_multi(self, sim, dcp)
        # hooks for custom writers written in Python, may be slow
        for f in sim.config.db.dcp_write_hooks:
            f(self, sim, dcp)
        self.commit()

    def save_final(self: Database, sim: SimulationEngine):
        sim.final_write_to_sqlite(self._connection)

    def dataframe(
        self, query: str, params: list | tuple | dict | None = None, dtype=None
    ):
        """Run a SQL query and return the results as a pandas DataFrame."""
        if not self.is_open:
            raise ValueError("database is not open")
        import pandas as pd

        return pd.read_sql_query(query, self._connection, params=params, dtype=dtype)

    def schema(self, name: str):
        """Get a table schema from the SQLite database."""
        if not self.is_open:
            raise ValueError("database is not open")
        try:
            return next(
                self.execute("SELECT sql FROM sqlite_master WHERE name = ?1", (name,))
            )[0]
        except Exception:
            raise

    def save_dataframe(
        self,
        name: str,
        df: pd.DataFrame,
        if_exists: Literal["fail", "replace", "append"] = "replace",
    ):
        """Save a dataframe as a table in this database."""
        df.to_sql(name, self._connection, if_exists=if_exists)

    def table_names(self) -> list[str]:
        """List of all tables in the database."""
        qry = "SELECT name FROM sqlite_master WHERE type=='table'"
        return [i[0] for i in self._connection.execute(qry)]

    def table_info(self, table_name: str) -> pd.DataFrame:
        """Get info about a table"""
        df = self.dataframe(f"PRAGMA table_info({table_name})")
        return df.set_index("cid")

    def index_names(self, table_name) -> list[str]:
        """List of all named indexes on a given table."""
        qry = "SELECT name FROM sqlite_master WHERE type=='index' AND tbl_name==?1"
        return [i[0] for i in self._connection.execute(qry, (table_name,))]

    def add_indexes(self, fare_detail=True, leg_detail=True):
        any_work = False
        if fare_detail and "fare_detail_idx_2" not in self.index_names("fare_detail"):
            logger.info("adding index on fare_detail")
            idx = """
            CREATE INDEX fare_detail_idx_2
            ON fare_detail (
                fare_id, scenario, trial, sample, days_prior
            );
            """
            self._connection.execute(idx)
            self._connection.commit()
            self._connection.execute("BEGIN TRANSACTION;")
            any_work = True

        if leg_detail and "leg_detail_idx_2" not in self.index_names("leg_detail"):
            logger.info("adding index on leg_detail")
            idx = """
            CREATE INDEX leg_detail_idx_2
            ON leg_detail (scenario, trial, sample, days_prior, flt_no);
            """
            self._connection.execute(idx)
            self._connection.commit()
            self._connection.execute("BEGIN TRANSACTION;")
            any_work = True

        if any_work:
            logger.info("completed adding indexes")

    def backup(self, dst: Path | str | sqlite3.Connection, show_progress: bool = True):
        """Back up this database to another copy."""
        if self.engine != "sqlite":
            raise NotImplementedError(f"no backup available for engine={self.engine!r}")
        if not self.is_open:
            raise OSError("database connection is not open")

        def _progress(status, remaining, total):
            if remaining:
                print(f"Copied {total - remaining} of {total} pages...")
            else:
                print(f"Copied all {total} pages.")

        if not isinstance(dst, sqlite3.Connection):
            dst = sqlite3.connect(dst)
        if self._connection.in_transaction:
            self._connection.execute("COMMIT;")
        with dst:
            self._connection.backup(
                dst, pages=10000, progress=_progress if show_progress else None
            )
        self._connection.execute("BEGIN TRANSACTION;")
        dst.close()


def get_database_connection(
    engine: Literal["sqlite", None] = "sqlite",
    filename: Path = None,
    pragmas: Iterable[str] = (),
    commit_count_delay: int | None = 250,
):
    return Database(
        engine=engine,
        filename=filename,
        pragmas=pragmas,
        commit_count_delay=commit_count_delay,
    )


def compute_rrd(sim: SimulationEngine, dep_time: float):
    tmp = int(dep_time / 86400) * 86400
    days_prior = int((tmp - sim.last_event_time) / 86400)
    if sim.num_events() == 0:
        days_prior = 0
    return days_prior


def delete_experiment(cnx: Database, name):
    with cnx:
        logger.info(f"deleting existing scenario {name!r} from database")
        cnx.execute(f"DELETE FROM leg_detail WHERE scenario = '{name}' ")
        cnx.execute(f"DELETE FROM leg_bucket_detail WHERE scenario = '{name}' ")
        cnx.execute(f"DELETE FROM demand_detail WHERE scenario = '{name}' ")
        cnx.execute(f"DELETE FROM fare_detail WHERE scenario = '{name}' ")


n_commit = 0


def sql_placeholders(cnx, n: int):
    if isinstance(cnx, Database):
        return sql_placeholders(cnx._connection, n)
    elif isinstance(cnx, sqlite3.Connection):
        x = "?"
    else:
        x = "%s"
    return ", ".join(x for _ in range(n))


# TODO - How to model RRD / capture date?
def save_leg(cnx, sim, leg, dcp) -> string:
    _dep_time = datetime.utcfromtimestamp(leg.dep_time).strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor = cnx.cursor()
        sql = f"""INSERT INTO leg_detail
                (scenario, iteration, trial, sample, days_prior, flt_no, sold, revenue)
                VALUES ({sql_placeholders(cnx, 8)})"""
        cursor.execute(
            sql,
            (
                sim.name,
                sim.iteration,
                sim.trial,
                sim.sample,
                dcp,
                leg.flt_no,
                leg.sold,
                leg.revenue,
            ),
        )
        return True
    except Exception as err:
        print(f"Doh !!! leg_detail: {err}")
        return False


leg_bucket_sql = {}


def save_leg_bucket_multi(
    cnx: Database, sim: SimulationEngine, leg, dcp, commit=False
) -> string:
    try:
        cursor = cnx.cursor()
        cnx_type = type(cnx).__name__
        if cnx_type not in leg_bucket_sql:
            sql = leg_bucket_sql[cnx_type] = f"""INSERT INTO leg_bucket_detail
                (scenario, iteration, trial, sample, days_prior, flt_no,
                bucket_number, name, auth, revenue, sold, untruncated_demand,
                forecast_mean) VALUES ({sql_placeholders(cnx, 13)})"""
        else:
            sql = leg_bucket_sql.get(cnx_type)
        data_list = []
        for n, bkt in enumerate(leg.buckets):
            data = (
                sim.name,
                sim.iteration,
                sim.trial,
                sim.sample,
                dcp,
                leg.flt_no,
                n,
                bkt.name,
                bkt.alloc,
                bkt.revenue,
                bkt.sold,
                bkt.untruncated_demand,
                bkt.fcst_mean,
            )
            data_list.append(data)

        cursor.executemany(sql, data_list)
        if commit:
            cnx.commit()
        cursor.close()
        return True
    except Exception as err:
        print(f"Doh !!! leg_bucket_detail: {err}")
        return False


def save_demand_multi(cnx: Database, sim: SimulationEngine, dcp) -> string:
    data_list = []
    for dmd in sim.demands:
        data_list.append(
            (
                sim.name,
                sim.iteration,
                sim.trial,
                sim.sample,
                dcp,
                dmd.orig,
                dmd.dest,
                dmd.segment,
                dmd.scenario_demand,
                dmd.sold,
                dmd.revenue,
            )
        )
        # if dmd.sold > dmd.scenario_demand:
        #     print(f"{dmd.orig=}, {dmd.dest=}, {dmd.segment},
        #     {dmd.sold}, {dmd.scenario_demand}")

    try:
        cursor = cnx.cursor()
        sql = f"""INSERT INTO demand_detail
                (scenario, iteration, trial, sample, days_prior,
                 orig, dest, segment, sample_demand, sold, revenue)
                VALUES ({sql_placeholders(cnx, 11)})"""
        cursor.executemany(sql, data_list)
        return True
    except Exception as err:
        print(f"Doh !!! demand_detail: {err}")
        return False


def save_fare_multi(cnx: Database, sim: SimulationEngine, dcp) -> string:
    data_list = []
    for fare in sim.fares:
        data_list.append(
            (
                sim.name,
                sim.iteration,
                sim.trial,
                sim.sample,
                dcp,
                fare.sold,
                fare.sold_business,
                fare.fare_id,
            )
        )
    try:
        cursor = cnx.cursor()
        sql = f"""INSERT INTO fare_detail
                (scenario, iteration, trial, sample, days_prior,
                 sold, sold_business, fare_id)
                VALUES ({sql_placeholders(cnx, 8)})"""
        cursor.executemany(sql, data_list)
        return True
    except Exception as err:
        print(f"Doh !!! fare: {err}")
        return False


#
# def get_legs(cnx: Database, _scenario="PODS 1 Leg", _trial=0, _rrd=0):
#     """Return a collection of legs
#        Mostly used by the airline part of the simulation code"""
#     legs = []
#     sql = """SELECT a.iteration, a.trial, a.sample, a.days_prior,
#                    a.carrier, a.orig, a.dest, a.flt_no,
#                    a.capacity, a.sold, a.q_demand, a.untruncated_demand,
#                    b.bucket_number, b.name AS bucket_name, b.auth, b.sold, b.revenue
#             FROM leg_detail a
#             JOIN leg_bucket_detail b USING(
#               iteration, trial, sample, days_prior, carrier, orig, dest, flt_no
#             )
#             WHERE a.scenario = %s
#               AND a.trial = %s
#               AND a.days_prior = %s
#             ORDER BY a.iteration, a.trial, a.sample, a.days_prior, a.carrier,
#                      a.orig, a.dest, a.flt_no, b.bucket_number;"""
#     try:
#         cursor = cnx.cursor()
#         cursor.execute(sql, (_scenario, _trial, _rrd))
#         old_key = None
#         for iteration, trial, sample, days_prior, carrier, orig, dest, flt_no, \
#                 capacity, sold, q_demand, untruncated_demand, \
#                 bucket_number, bucket_name, auth, sold, bkt_revenue in cursor:
#             key = (iteration, trial, sample, days_prior, carrier, orig, dest, flt_no)
#             if key != old_key:
#                 tmp_leg = Leg(carrier, flt_no, orig, dest, capacity=capacity)
#                 legs.append(tmp_leg)
#             bkt = Bucket(bucket_name, sold=sold)
#             bkt.revenue = bkt_revenue
#             tmp_leg.add_bucket(bkt)
#             old_key = key
#     except Exception as err:
#         print("Doh !!! get_legs: {}".format(err))
#         return []
#
#     return legs
