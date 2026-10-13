#
# Utilities for reading and writing PassengerSim data
# Uses SQLITE
#
from __future__ import annotations

import logging
import math
import sqlite3
import string
import warnings
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import pandas as pd
from passengersim_core import DbWriter

from passengersim.config import Config
from passengersim.core import SimulationEngine
from passengersim.utils._database_compat import deprecated_cnx_keyword

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


class VoidWriter:
    """A dummy DbWriter that does nothing."""

    def close(self):
        pass

    def update_db_write_flags(self):
        pass

    def commit(self):
        pass

    def write_to_sqlite(self, *args, **kwargs):
        pass

    def final_write_to_sqlite(self, *args, **kwargs):
        pass


class Database:
    """A wrapper to manage transactions for PassengerSim on SQLite."""

    _db_writer: DbWriter | VoidWriter | None

    def __init__(
        self,
        engine: Literal["sqlite", None] = "sqlite",
        filename=None,
        pragmas: Iterable[str] = (),
        commit_count_delay: int | None = 250,
        *,
        connection: sqlite3.Connection | None = None,
    ):
        if isinstance(engine, str) and engine.endswith(".sqlite") and filename is None:
            filename = engine
            engine = "sqlite"
        if connection is not None and not isinstance(connection, sqlite3.Connection):
            raise TypeError("connection must be a sqlite3.Connection")
        if connection is not None and (engine != "sqlite" or filename is not None):
            raise ValueError("connection cannot be combined with a filename or non-SQLite engine")
        self._connection = connection
        self._owns_connection = connection is None
        self._db_writer = None
        self._contexts = []
        self.engine = engine
        self.filename = str(filename) if filename is not None else None
        self.pragmas = pragmas
        self._counter = 0
        self._commit_count_delay = commit_count_delay
        if self._commit_count_delay is not None:
            self.commit = self._commit_by_count
        else:
            self.commit = self._commit_raw
        if connection is None:
            self.open()
        else:
            self._initialize_connection()

    def __del__(self):
        # Cleanup never commits and never closes a caller-owned connection.
        try:
            if self.__dict__.get("_owns_connection", False):
                connection = self.__dict__.get("_connection")
                if connection is not None:
                    connection.close()
        except Exception:
            pass

    @property
    def database_path(self):
        """Absolute main-database path, or None for memory/disabled connections."""
        if self.filename is not None:
            return Path(self.filename).absolute() if self.filename != ":memory:" else None
        if self._connection is not None:
            for _, name, path in self._connection.execute("PRAGMA database_list"):
                if name == "main" and path:
                    return Path(path).absolute()
        return None

    @property
    def connection(self) -> sqlite3.Connection | None:
        """The Python connection shared with the native writer."""
        return self._connection

    def connect_to_simulation_engine(self, sim: SimulationEngine):
        """Attach the native writer to this connection without reopening it."""
        if self._db_writer is None:
            if self._connection is not None:
                self._db_writer = DbWriter("db", sim.config, sim, self._connection)
            else:
                self._db_writer = VoidWriter()

    def disconnect_from_simulation_engine(self):
        """Detach the writer while leaving the Python connection open."""
        if self._db_writer:
            self._db_writer.close()
            self._db_writer = None

    def __getattr__(self, item):
        return getattr(self._connection, item)

    def open(self, filename: str | None = None):
        """Open the connection if it is not already open."""
        if not self._owns_connection:
            raise ConnectionError("a caller-owned connection cannot be reopened by Database")
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
            self._initialize_connection()
        else:
            raise NotImplementedError(f"unknown engine {self.engine!r}")

    def _initialize_connection(self):
        """Initialize schema and functions without committing borrowed work."""
        from .tables import create_tables

        self._connection.create_aggregate("VARIANCE", 1, _VarianceFunc)
        self._connection.create_aggregate("STDEV", 1, _StdevFunc)
        for pragma in self.pragmas:
            self._connection.execute(f"PRAGMA {pragma};")
        if not self._connection.in_transaction:
            self._connection.execute("BEGIN TRANSACTION;")
        create_tables(self)

    def close(self):
        """Detach the writer; commit and close only a library-owned connection."""
        self.disconnect_from_simulation_engine()
        if self._connection is not None:
            if self._owns_connection:
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
        if self._connection and self._owns_connection and not self._contexts:
            if self._connection.in_transaction:
                self._connection.execute("COMMIT;")
            self._connection.execute("BEGIN TRANSACTION;")

    def __enter__(self):
        if not self.is_open:
            raise ConnectionError("the database connection is closed")
        # Savepoints keep contexts nestable and preserve borrowed transactions.
        if not self._connection.in_transaction:
            self._connection.execute("BEGIN")
        name = f"passengersim_context_{len(self._contexts)}"
        self._connection.execute(f"SAVEPOINT {name}")
        self._contexts.append(name)
        return self._connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Release this scope without hiding an exception from the context body.

        Parameters
        ----------
        exc_type : type of BaseException or None
            Exception type raised by the context body, if any.
        exc_val : BaseException or None
            Original exception, which takes precedence over cleanup failures.
        exc_tb : types.TracebackType or None
            Original exception traceback.

        Returns
        -------
        False
            Propagate the body's exception, if any.

        Raises
        ------
        sqlite3.Error
            Cleanup fails when there is no original exception to propagate.
        """
        name = self._contexts.pop()
        try:
            # SQLite may roll back the entire transaction (for example, after
            # INSERT OR ROLLBACK), removing all nested savepoints at once.
            if self._connection.in_transaction:
                if exc_type is not None:
                    self._connection.execute(f"ROLLBACK TO {name}")
                self._connection.execute(f"RELEASE {name}")
            if not self._contexts and self._owns_connection:
                self._commit_raw()
        except sqlite3.Error as cleanup_error:
            if exc_val is None:
                raise
            exc_val.add_note(f"Database context cleanup also failed: {cleanup_error}")
        return False

    @property
    def is_open(self) -> bool:
        return self._connection is not None

    def use_sqlite(self) -> bool:
        """Return True when this wrapper holds a SQLite connection."""
        return self.engine == "sqlite" and self._connection is not None

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
                cfg.model_dump_json(exclude={"raw_license_certificate": True}),
            ),
        )

    def load_raw_configs(self, scenario=None) -> dict | Any:
        import json

        if scenario:
            rawjson = next(
                self.execute(
                    "SELECT configs, max(updated_at) FROM runtime_configs WHERE scenario = ?1",
                    (scenario,),
                )
            )[0]
        else:
            rawjson = next(self.execute("SELECT configs, max(updated_at) FROM runtime_configs"))[0]
        result = json.loads(rawjson)
        if not isinstance(result, dict):
            warnings.warn("malformed configs, not a mapping", stacklevel=2)
        return result

    def load_configs(self, scenario=None, on_validation_error: Literal["raise", "ignore"] = "raise") -> Config | Any:
        raw = self.load_raw_configs(scenario)

        from pydantic import ValidationError

        try:
            return Config.model_validate(raw)
        except ValidationError as err:
            logger.error(f"error loading configs: {err}")
            if on_validation_error == "raise":
                raise
            else:
                return raw

    def save_details(self: Database, db_writer: DbWriter, sim: SimulationEngine, dcp: int):
        """
        Save details, can be done at each RRD/DCP and at the end of the run
        """
        if not sim.save_timeframe_details and dcp > 0:
            return
        if sim.config.db.fast and isinstance(self._connection, sqlite3.Connection):
            _internal_log = db_writer.write_to_sqlite(
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
        self.commit()

    def save_final(self: Database, sim: SimulationEngine):
        sim.final_write_to_sqlite(self._connection)

    def dataframe(self, query: str, params: list | tuple | dict | None = None, dtype=None):
        """Run a SQL query and return the results as a pandas DataFrame."""
        with self:
            return pd.read_sql_query(query, self._connection, params=params, dtype=dtype)

    def schema(self, name: str):
        """Get a table schema from the SQLite database."""
        if not self.is_open:
            raise ValueError("database is not open")
        try:
            return next(self.execute("SELECT sql FROM sqlite_master WHERE name = ?1", (name,)))[0]
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
            self._commit_raw()
            any_work = True

        if leg_detail and "leg_detail_idx_2" not in self.index_names("leg_detail"):
            logger.info("adding index on leg_detail")
            idx = """
            CREATE INDEX leg_detail_idx_2
            ON leg_detail (scenario, trial, sample, days_prior, leg_id);
            """
            self._connection.execute(idx)
            self._commit_raw()
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

        if not self._owns_connection and self._connection.in_transaction:
            raise ValueError("Commit or roll back the caller-owned connection before backup")
        owns_destination = not isinstance(dst, sqlite3.Connection)
        if not owns_destination and dst.in_transaction:
            raise ValueError("Commit or roll back the destination connection before backup")
        if owns_destination:
            dst = sqlite3.connect(dst)
        try:
            if self._connection.in_transaction:
                self._connection.execute("COMMIT;")
            self._connection.backup(dst, pages=10000, progress=_progress if show_progress else None)
        finally:
            if self._owns_connection:
                self._connection.execute("BEGIN TRANSACTION;")
            if owns_destination:
                dst.close()


def get_database_connection(
    engine: Literal["sqlite", None] = "sqlite",
    filename: Path = None,
    pragmas: Iterable[str] = (),
    commit_count_delay: int | None = 250,
):
    """Return a PassengerSim Database wrapper, not a raw SQLite connection."""
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


@deprecated_cnx_keyword()
def delete_experiment(db: Database, name):
    with db:
        logger.info(f"deleting existing scenario {name!r} from database")
        db.execute(f"DELETE FROM leg_detail WHERE scenario = '{name}' ")
        db.execute(f"DELETE FROM leg_bucket_detail WHERE scenario = '{name}' ")
        db.execute(f"DELETE FROM demand_detail WHERE scenario = '{name}' ")
        db.execute(f"DELETE FROM fare_detail WHERE scenario = '{name}' ")


n_commit = 0


@deprecated_cnx_keyword()
def sql_placeholders(db: Database | sqlite3.Connection, n: int):
    if isinstance(db, Database):
        return sql_placeholders(db._connection, n)
    elif isinstance(db, sqlite3.Connection):
        x = "?"
    else:
        x = "%s"
    return ", ".join(x for _ in range(n))


# TODO - How to model RRD / capture date?
@deprecated_cnx_keyword()
def save_leg(db: Database | sqlite3.Connection, sim, leg, dcp) -> string:
    _dep_time = datetime.utcfromtimestamp(leg.dep_time).strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor = db.cursor()
        sql = f"""INSERT INTO leg_detail
                (scenario, iteration, trial, sample, days_prior, leg_id, sold, revenue)
                VALUES ({sql_placeholders(db, 8)})"""
        cursor.execute(
            sql,
            (
                sim.name,
                sim.iteration,
                sim.trial,
                sim.sample,
                dcp,
                leg.leg_id,
                leg.sold,
                leg.revenue,
            ),
        )
        return True
    except Exception as err:
        print(f"Doh !!! leg_detail: {err}")
        return False


leg_bucket_sql = {}


@deprecated_cnx_keyword()
def save_leg_bucket_multi(db: Database, sim: SimulationEngine, leg, dcp, commit=False) -> string:
    try:
        cursor = db.cursor()
        db_type = type(db).__name__
        if db_type not in leg_bucket_sql:
            sql = leg_bucket_sql[db_type] = f"""INSERT INTO leg_bucket_detail
                (scenario, iteration, trial, sample, days_prior, leg_id,
                bucket_number, name, auth, revenue, sold, detruncated_demand,
                forecast_mean) VALUES ({sql_placeholders(db, 13)})"""
        else:
            sql = leg_bucket_sql.get(db_type)
        data_list = []
        for n, bkt in enumerate(leg.buckets):
            data = (
                sim.name,
                sim.iteration,
                sim.trial,
                sim.sample,
                dcp,
                leg.leg_id,
                n,
                bkt.name,
                bkt.alloc,
                bkt.revenue,
                bkt.sold,
                bkt.detruncated_demand,
                bkt.fcst_mean,
            )
            data_list.append(data)

        cursor.executemany(sql, data_list)
        if commit:
            db.commit()
        cursor.close()
        return True
    except Exception as err:
        print(f"Doh !!! leg_bucket_detail: {err}")
        return False


@deprecated_cnx_keyword()
def save_demand_multi(db: Database, sim: SimulationEngine, dcp) -> string:
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
        cursor = db.cursor()
        sql = f"""INSERT INTO demand_detail
                (scenario, iteration, trial, sample, days_prior,
                 orig, dest, segment, sample_demand, sold, revenue)
                VALUES ({sql_placeholders(db, 11)})"""
        cursor.executemany(sql, data_list)
        return True
    except Exception as err:
        print(f"Doh !!! demand_detail: {err}")
        return False


@deprecated_cnx_keyword()
def save_fare_multi(db: Database, sim: SimulationEngine, dcp) -> string:
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
        cursor = db.cursor()
        sql = f"""INSERT INTO fare_detail
                (scenario, iteration, trial, sample, days_prior,
                 sold, sold_business, fare_id)
                VALUES ({sql_placeholders(db, 8)})"""
        cursor.executemany(sql, data_list)
        return True
    except Exception as err:
        print(f"Doh !!! fare: {err}")
        return False
