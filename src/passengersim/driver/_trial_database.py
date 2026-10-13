"""Parent-owned SQLite files and the small, picklable worker handoff."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from passengersim.config import Config
    from passengersim.database import Database

# A fork must never let a child's tempfile finalizer delete the parent's files.
_STORES = weakref.WeakSet()


def _after_fork():
    for store in list(_STORES):
        if getattr(store, "_temporary", None) is not None:
            store._temporary._finalizer.detach()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def finalize_result(result):
    """Bind a completed result's connection on the consuming thread."""
    store = getattr(result, "_trial_database_store", None)
    if store is not None:
        _ = store.database
    return result


@dataclass(frozen=True)
class TrialDatabase:
    """A committed, closed main database produced by one successful trial."""

    path: Path
    trial_id: int
    schema_version: int
    user_version: int
    sqlite_version: str
    complete: bool = True


@dataclass
class TrialResult:
    """Return summary and database descriptor once, through the Future."""

    summary: Any
    database: TrialDatabase | None
    timings: dict[str, float] = field(default_factory=dict)
    published_at: float = 0.0


class TrialDatabaseStore:
    """Own one run's directory in the parent; never send this owner to workers."""

    def __init__(
        self,
        config: Config,
        initializer: Callable[[sqlite3.Connection], None] | None = None,
        *,
        standalone: bool = False,
    ) -> None:
        self._pid = os.getpid()
        self._standalone = standalone
        self._thread = None
        self._initializer = initializer
        self._running = False
        _STORES.add(self)
        self._temporary = None
        self.directory = None
        self.completed = {}
        self.metrics = {}
        self.merged = set()
        self.failed = {}
        self.cancelled = set()
        self.expected = set(range(config.simulation_controls.num_trials))
        self.finalized = False
        self.interrupted = False
        self._database = None
        self._closed = False
        self._closing = False
        self._pending = set()
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        from passengersim.config.storage import resolve_storage, validate_database_reports

        db = config.db
        self.policy = resolve_storage(db, orchestrated=not standalone)
        self._db_config = db if standalone else None
        self._path = None
        validate_database_reports(config, self.policy)
        if self.policy.mode in {"disabled", "memory", "summary_only"}:
            return
        root = self.policy.directory
        if self.policy.filename is not None:
            # Reserve exact destinations exclusively: never append to/overwrite an
            # earlier run. The parent's consolidated file differs from trial files.
            self.policy.filename.parent.mkdir(parents=True, exist_ok=True)
            self.policy.filename.touch(exist_ok=False)
            self._path = self.policy.filename
            root = self.policy.filename.parent
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        if self.policy.mode == "temporary":
            self._temporary = tempfile.TemporaryDirectory(prefix="passengersim-db-", dir=root)
            # Weakref finalizers run before cyclic-GC destructors. Let this owner
            # close SQLite first instead of racing tempfile's independent cleanup.
            self._temporary._finalizer.detach()
            self.directory = Path(self._temporary.name).resolve()
        elif standalone and self._path is not None:
            self.directory = self._path.parent
        else:
            self.directory = Path(tempfile.mkdtemp(prefix="passengersim-db-", dir=root)).resolve()
        if self._path is None:
            self._path = self.directory / "results.sqlite"

    @property
    def consolidated_path(self):
        return self._path

    @property
    def available(self) -> bool:
        """Report availability without opening a lazy merged connection."""
        if self._closed or not self.finalized or (not self.merged and not self._standalone):
            return False
        if self.policy.mode in {"memory", "summary_only"}:
            return self._database is not None and self._database._connection is not None
        return self._path is not None and self._path.is_file()

    @property
    def status(self):
        """Return a snapshot of trial outcomes, including incomplete runs."""
        return {
            "completed": sorted(self.merged),
            "failed": dict(self.failed),
            "cancelled": sorted(self.cancelled),
            "complete": self.finalized and not self.interrupted and self.merged == self.expected,
        }

    def check_access(self):
        """Reject inherited handles before touching locks or SQLite."""
        if os.getpid() != self._pid:
            raise RuntimeError("A retained database cannot be used or closed in an inherited process")
        if self._thread is not None and threading.get_ident() != self._thread:
            raise sqlite3.ProgrammingError("Retained database belongs to another thread")

    @property
    def database(self) -> Database | None:
        """Open the finalized result on first access in the consuming thread."""
        self.check_access()
        if (
            self._closed
            or not self.finalized
            or (not self.merged and not self._standalone)
            or self.policy.mode == "disabled"
            or (self.policy.mode == "summary_only" and not self._standalone)
        ):
            return None
        with self._lock:
            self.check_access()
            if self._database is None:
                from passengersim.database._retained import RetainedConnection, RetainedDatabase

                self._thread = threading.get_ident()
                connection = sqlite3.connect(self.consolidated_path or ":memory:", factory=RetainedConnection)
                connection._result_store = self
                try:
                    database = RetainedDatabase(
                        connection=connection,
                        pragmas=self._db_config.pragmas if self._db_config is not None else (),
                        commit_count_delay=self._db_config.commit_count_delay if self._db_config is not None else 250,
                    )
                    database._owns_connection = True
                    database.filename = (
                        str(self.consolidated_path) if self.consolidated_path is not None else ":memory:"
                    )
                    database._result_store = self
                    if self._initializer is not None:
                        self._initializer(connection)
                    self._database = database
                except BaseException:
                    connection.close()
                    self._thread = None
                    raise
            return self._database

    def accept(self, trial_id, value):
        """Measure parent receipt and merge without transporting database contents."""
        started = time.perf_counter()
        metrics = dict(value.timings)
        metrics["receive_delay_s"] = max(0.0, started - value.published_at) if value.published_at else 0.0
        self.metrics[trial_id] = metrics
        try:
            if value.database is not None:
                if value.database.trial_id != trial_id:
                    raise ValueError("Worker returned the wrong trial identity")
                self.merge(value.database)
            else:
                self.merged.add(trial_id)
        finally:
            metrics["merge_s"] = time.perf_counter() - started

    def merge(self, descriptor):
        """Commit one complete trial atomically before releasing its transfer file."""
        trial_id = descriptor.trial_id
        if trial_id in self.merged:
            raise ValueError(f"Trial {trial_id} has already been merged")
        if trial_id not in self.expected or not descriptor.complete:
            raise ValueError(f"Invalid completed trial descriptor: {trial_id}")
        merge_trial(self.consolidated_path, descriptor)
        self.merged.add(trial_id)
        self.completed[trial_id] = descriptor
        if self._temporary is not None:
            # The attachment is closed and the target committed before unlinking.
            # Descriptors now identify the consolidated file; filter by trial_id.
            self.completed[trial_id] = replace(descriptor, path=self.consolidated_path)
            try:
                descriptor.path.unlink()
            except OSError as error:
                # A cleanup failure cannot undo an already committed merge.
                logging.getLogger(__name__).warning(
                    "Merged trial %s, but could not remove %s: %s", trial_id, descriptor.path, error
                )

    def path_for(self, trial_id):
        """Allocate a trial filename without opening a parent connection."""
        if self.directory is None:
            return None
        return self.directory / f"trial-{trial_id:04d}.sqlite"

    def track(self, future):
        """Keep storage alive until a worker Future finishes, including failure."""
        with self._lock:
            self._pending.add(future)
            self._idle.clear()
        future.add_done_callback(self._finished)

    def _finished(self, future):
        """Release the pending reference after the worker stops using its files."""
        with self._lock:
            self._pending.discard(future)
            if not self._pending:
                self._idle.set()

    def wait(self):
        """Wait for descriptor callbacks as well as Future completion."""
        self._idle.wait()

    def close(self):
        """Invalidate shared aliases and close handles before deleting temporary files."""
        self.check_access()
        with self._lock:
            if self._pending or self._running:
                raise RuntimeError("Cannot close result databases while workers or the merger are active")
            if self._database is not None and not self._closed:
                from passengersim.database import Database

                self._closing = True
                try:
                    Database.close(self._database)
                finally:
                    self._closing = False
            self._closed = True
            if self._temporary is not None:
                # Keep the owner on failure so an explicit close can retry.
                self._temporary.cleanup()

    def __del__(self):
        try:
            if os.getpid() != self._pid:
                return
            # Cyclic GC can keep the connection alive after dropping our reference.
            # Close it explicitly before unlinking files (required on Windows),
            # bypassing shared-owner close so finalization never commits or recurses.
            if self._database is not None:
                connection = self._database._connection
                if connection is not None:
                    sqlite3.Connection.close(connection)
                    self._database._connection = None
            if self._temporary is not None:
                self._temporary.cleanup()
        except Exception:
            pass


def finish_database(database, trial_id):
    """Commit/checkpoint a worker database and close it before publishing its path."""
    connection = database.connection
    if connection is None:
        return None
    try:
        database.disconnect_from_simulation_engine()
        connection.commit()
        if connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal":
            busy, _, _ = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if busy:
                raise RuntimeError("Cannot publish trial database: WAL checkpoint is busy")
        descriptor = TrialDatabase(
            Path(database.filename).resolve(),
            trial_id,
            connection.execute("PRAGMA schema_version").fetchone()[0],
            connection.execute("PRAGMA user_version").fetchone()[0],
            sqlite3.sqlite_version,
        )
    finally:
        database.close()
    return descriptor


# Policies are deliberately exhaustive. Report cache tables are derived from
# detail rows and must be recomputed across trials, never copied from one worker.
_STATIC_TABLES = {"leg_defs", "fare_defs", "fare_restriction_defs", "path_defs", "booking_curve", "distance"}
_TRIAL_TABLES = {
    "leg_detail",
    "leg_bucket_detail",
    "demand_detail",
    "fare_detail",
    "bookings_by_timeframe",
    "path_class_detail",
    "edgar",
}
_REPORT_CACHES = {
    "fare_summary",
    "demand_to_come_summary",
    "bid_price_general_summary",
    "bid_price_somecap_summary",
    "displacement_summary",
}


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _configs(connection, schema):
    """Compare configuration content while ignoring per-worker file paths/times."""
    rows = []
    for scenario, version, raw in connection.execute(
        f"SELECT scenario, pxsim_version, configs FROM {schema}.runtime_configs"
    ):
        cfg = json.loads(raw)
        if cfg.get("db"):
            cfg["db"].pop("filename", None)
            if "write_items" in cfg["db"]:
                cfg["db"]["write_items"] = sorted(cfg["db"]["write_items"])
        if cfg.get("outputs") and "reports" in cfg["outputs"]:
            cfg["outputs"]["reports"] = sorted(cfg["outputs"]["reports"], key=lambda item: json.dumps(item))
        rows.append((scenario, version, json.dumps(cfg, sort_keys=True)))
    return sorted(rows)


def merge_trial(target, descriptor):
    """Merge known tables from one closed worker file in a single transaction.

    Parameters
    ----------
    target : pathlib.Path
        Parent-owned consolidated database path.
    descriptor : TrialDatabase
        Completed worker database, opened read-only during the merge.

    Raises
    ------
    ValueError
        Unknown schema objects, conflicting definitions, or invalid trial IDs.
    sqlite3.Error
        A database error; all changes for this trial are rolled back.
    """
    connection = sqlite3.connect(target, uri=True)
    try:
        connection.execute("ATTACH DATABASE ? AS worker", (descriptor.path.as_uri() + "?mode=ro",))
        connection.execute("BEGIN")
        tables = dict(
            connection.execute(
                "SELECT name, sql FROM worker.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        )
        unknown = set(tables) - _STATIC_TABLES - _TRIAL_TABLES - _REPORT_CACHES - {"runtime_configs"}
        unsupported = connection.execute(
            "SELECT name FROM worker.sqlite_master WHERE type IN ('view', 'trigger')"
        ).fetchall()
        if unknown or unsupported:
            raise ValueError(
                f"No merge policy for user schema objects: {sorted(unknown)} {unsupported}. "
                "Use db.result_storage='summary_only' until an explicit policy is provided."
            )
        existing = dict(
            connection.execute(
                "SELECT name, sql FROM main.sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        )
        supported = set(tables) - _REPORT_CACHES
        if existing and set(existing) != supported:
            raise ValueError("Trial database table sets differ")
        indexes = {}
        for schema in ("main", "worker"):
            indexes[schema] = {
                name: sql
                for name, table, sql in connection.execute(
                    f"SELECT name, tbl_name, sql FROM {schema}.sqlite_master WHERE type='index'"
                )
                if table in supported
            }
        if existing and indexes["main"] != indexes["worker"]:
            raise ValueError("Trial database indexes differ")
        version = connection.execute("PRAGMA worker.user_version").fetchone()[0]
        if version != descriptor.user_version:
            raise ValueError("Trial database user_version changed after publication")
        if existing and version != connection.execute("PRAGMA main.user_version").fetchone()[0]:
            raise ValueError("Trial database user_version differs")
        connection.execute(f"PRAGMA main.user_version={int(version)}")
        fare_map = {}
        if "fare_defs" in supported:
            # Fare IDs come from a process-global counter. Match full definitions
            # and multiplicities, then translate detail foreign keys to the first
            # successful worker's IDs (also handles repeated identical fares).
            groups = []
            for schema in ("main", "worker") if "fare_defs" in existing else ("worker", "worker"):
                definitions = {}
                for row in connection.execute(f"SELECT * FROM {schema}.fare_defs ORDER BY fare_id"):
                    definitions.setdefault(row[1:], []).append(row[0])
                groups.append(definitions)
            if groups[0].keys() != groups[1].keys() or any(
                len(groups[0][key]) != len(groups[1][key]) for key in groups[0]
            ):
                raise ValueError("Conflicting static definitions in fare_defs")
            for key, ids in groups[1].items():
                fare_map.update(zip(ids, groups[0][key], strict=True))
            connection.execute("CREATE TEMP TABLE fare_id_map (source INTEGER PRIMARY KEY, target INTEGER)")
            connection.executemany("INSERT INTO fare_id_map VALUES (?, ?)", fare_map.items())
        for table in sorted(supported):
            name = _quote(table)
            if table in existing and existing[table] != tables[table]:
                raise ValueError(f"Trial schema differs for {table}")
            if table not in existing:
                connection.execute(tables[table])
            if table in _TRIAL_TABLES:
                if connection.execute(
                    f"SELECT 1 FROM worker.{name} WHERE trial IS NOT ? LIMIT 1", (descriptor.trial_id,)
                ).fetchone():
                    raise ValueError(f"Unexpected trial ID in {table}")
                if connection.execute(
                    f"SELECT 1 FROM main.{name} WHERE trial = ? LIMIT 1", (descriptor.trial_id,)
                ).fetchone():
                    raise ValueError(f"Duplicate trial {descriptor.trial_id} in {table}")
            elif table in existing:
                if table == "fare_defs":
                    continue  # Validated in full while constructing the ID map.
                if table == "runtime_configs":
                    matches = _configs(connection, "main") == _configs(connection, "worker")
                else:
                    # Include multiplicity, NULLs, and every definition column.
                    cols = ", ".join(_quote(row[1]) for row in connection.execute(f"PRAGMA worker.table_info({name})"))
                    left = f"SELECT {cols}, count(*) FROM main.{name} GROUP BY {cols}"
                    right = f"SELECT {cols}, count(*) FROM worker.{name} GROUP BY {cols}"
                    matches = not connection.execute(f"{left} EXCEPT {right}").fetchone()
                    matches = matches and not connection.execute(f"{right} EXCEPT {left}").fetchone()
                if not matches:
                    raise ValueError(f"Conflicting static definitions in {table}")
                continue
            if table == "fare_detail" and "fare_defs" in supported:
                if connection.execute(
                    "SELECT 1 FROM worker.fare_detail d LEFT JOIN fare_id_map m ON d.fare_id=m.source "
                    "WHERE m.source IS NULL LIMIT 1"
                ).fetchone():
                    raise ValueError("Unknown fare_id in fare_detail")
                columns = [row[1] for row in connection.execute(f"PRAGMA worker.table_info({name})")]
                expressions = ", ".join("m.target" if col == "fare_id" else f"d.{_quote(col)}" for col in columns)
                connection.execute(
                    f"INSERT INTO main.{name} SELECT {expressions} FROM worker.{name} d "
                    "JOIN fare_id_map m ON d.fare_id=m.source"
                )
            else:
                connection.execute(f"INSERT INTO main.{name} SELECT * FROM worker.{name}")
        if not existing:
            for table, sql in connection.execute(
                "SELECT tbl_name, sql FROM worker.sqlite_master WHERE type='index' AND sql IS NOT NULL"
            ).fetchall():
                if table in supported:
                    connection.execute(sql)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        # Closing also detaches the read-only worker database on Windows.
        connection.close()
