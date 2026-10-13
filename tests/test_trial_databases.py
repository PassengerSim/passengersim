"""Completed worker files survive process exit without transporting connections."""

import concurrent.futures
import gc
import pickle
import queue
import sqlite3
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Never

import pytest

import passengersim as pax
from passengersim.config.database import DatabaseConfig
from passengersim.driver._multiproc import _subprocess_run_trial
from passengersim.driver._trial_database import TrialDatabaseStore

pytestmark = pytest.mark.filterwarnings("error:.*cnx.*:DeprecationWarning")


@pytest.fixture
def config(tmp_path):
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 3
    cfg.simulation_controls.burn_samples = 1
    cfg.simulation_controls.show_progress_bar = False
    cfg.outputs._write_no_files()
    cfg.outputs.reports.clear()
    cfg.db = DatabaseConfig(result_storage="temporary", result_directory=tmp_path)
    return cfg


@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize("wal", [False, True])
def test_completed_files_and_result_lifetime(config, sequential, wal):
    if wal:
        config.db.pragmas = ["journal_mode=WAL"]
    sim = pax.MultiSimulation(config)
    summary = sim.sequential_run() if sequential else sim.run(max_processes=2)
    assert summary.db is sim.db
    timings = summary._metadata.get("database.timings")
    if timings is not None:
        assert all(value >= 0 for trial in timings.values() for value in trial.values())
    assert summary.database_status == {"completed": [0, 1], "failed": {}, "cancelled": [], "complete": True}
    assert summary.trial_databases == sim.trial_databases
    assert set(summary.trial_databases) == {0, 1}
    paths = [value.path for value in summary.trial_databases.values()]
    assert paths[0] == paths[1]
    assert not list(paths[0].parent.glob("trial-*.sqlite"))
    for trial, descriptor in summary.trial_databases.items():
        assert descriptor.complete and descriptor.trial_id == trial
        assert descriptor.schema_version > 0
        assert descriptor.sqlite_version == sqlite3.sqlite_version
        assert descriptor.path.is_absolute()
        with sqlite3.connect(descriptor.path) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert connection.execute("SELECT DISTINCT trial FROM leg_detail ORDER BY trial").fetchall() == [(0,), (1,)]
            assert connection.execute("SELECT count(*) FROM leg_detail").fetchone()[0] > 0
        connection.close()
        assert not descriptor.path.with_name(descriptor.path.name + "-wal").exists()
    # Temporary ownership must survive dropping the simulation, but not persist
    # as a stale path in summary-only serialized caches.
    restored = pickle.loads(pickle.dumps(summary))
    assert restored.trial_databases == {}
    owner = weakref.ref(sim._trial_database_store)
    del sim
    gc.collect()
    assert owner() is not None and all(path.exists() for path in paths)
    del summary
    gc.collect()
    assert owner() is None
    assert all(not path.exists() for path in paths)


def test_worker_publishes_only_closed_path(config):
    store = TrialDatabaseStore(config)
    progress = queue.Queue()
    value = _subprocess_run_trial(0, config.model_dump_json(), progress, database_path=store.path_for(0))
    assert not isinstance(value, Exception), value
    assert value.summary.db is None
    assert pickle.loads(pickle.dumps(value.database)) == value.database
    messages = list(progress.queue)
    assert messages[-1] == (0, "done", None)
    assert all(isinstance(payload, (int, type(None))) for _, _, payload in messages)
    # Windows cannot rename an open SQLite file: this is also a handle test.
    renamed = value.database.path.with_suffix(".renamed")
    value.database.path.rename(renamed)
    store.close()
    assert not renamed.exists()


def test_policy_validation_and_summary_only(config):
    config.db.filename = ":memory:"
    with pytest.raises(ValueError, match="conflicts"):
        pax.MultiSimulation(config).sequential_run()
    config.db.result_storage = "summary_only"
    config.db.result_directory = None
    summary = pax.MultiSimulation(config).run(max_processes=2)
    assert summary.trial_databases == {}
    config.db.engine = None
    config.db.result_storage = "temporary"
    assert TrialDatabaseStore(config).directory is None


def test_persistent_files_and_distinct_runs(config):
    config.db.result_storage = "persistent"
    first = pax.MultiSimulation(config)
    second = pax.MultiSimulation(config)
    a, b = first.sequential_run(), second.sequential_run()
    assert a.trial_databases[0].path != b.trial_databases[0].path
    first._trial_database_store.close()
    assert all(item.path.exists() for item in a.trial_databases.values())


def test_storage_defaults_and_active_cleanup(config):
    config.db = DatabaseConfig(result_directory=config.db.result_directory)
    store = TrialDatabaseStore(config)
    assert store._temporary is not None
    future = concurrent.futures.Future()
    store.track(future)
    with pytest.raises(RuntimeError, match="active"):
        store.close()
    future.set_exception(RuntimeError("worker failed"))
    assert store.completed == {}
    store.close()
    config.db.filename = config.db.result_directory / "requested.sqlite"
    config.db.result_directory = None
    explicit = TrialDatabaseStore(config)
    assert explicit._temporary is None


def fail_first_trial(sim):
    if sim.eng.trial == 0:
        raise RuntimeError("intentional trial failure")


def test_partial_failure_publishes_only_successful_trials(config):
    sim = pax.MultiSimulation(config)
    sim.begin_sample_callback(fail_first_trial)
    summary = sim.run(max_processes=2, fail_fast=False)
    assert set(summary.trial_databases) == {1}
    assert summary.trial_databases[1].path.exists()


def test_default_policy_survives_json(config):
    config.db = DatabaseConfig(result_directory=config.db.result_directory)
    restored = pax.Config.model_validate_json(config.model_dump_json())
    store = TrialDatabaseStore(restored)
    assert store._temporary is not None
    store.close()


@pytest.mark.parametrize("retain_sims", [True, False])
@pytest.mark.parametrize("single_process", [True, False])
def test_experiments_share_file_policy(config, tmp_path, single_process, retain_sims):
    from passengersim.experiments import Experiments

    experiments = Experiments(config, output_dir=tmp_path / "experiments")

    @experiments
    def first(cfg):
        return cfg

    @experiments
    def second(cfg):
        return cfg

    results = experiments.run(
        use_existing=False,
        single_process=single_process,
        retain_sims=retain_sims,
        write_report=False,
        cache_results=False,
    )
    if retain_sims:
        assert experiments.sims["first"].db is results["first"].db
        assert experiments.sims["second"].db is results["second"].db
    assert results["first"].db is not results["second"].db
    results["first"].db.execute("CREATE TABLE isolation_marker (value)")
    assert results["second"].db.execute("SELECT name FROM sqlite_master WHERE name='isolation_marker'").fetchall() == []
    assert set(results["first"].trial_databases) == {0, 1}
    assert set(results["second"].trial_databases) == {0, 1}
    assert results["first"].trial_databases[0].path != results["second"].trial_databases[0].path
    assert all(item.path.exists() for result in results.values() for item in result.trial_databases.values())


def test_finalization_failure_never_publishes_file(config, monkeypatch):
    import passengersim.driver._multiproc as driver

    def fail_checkpoint(*args):
        raise OSError("disk full during finalization")

    monkeypatch.setattr(driver, "finish_database", fail_checkpoint)
    store = TrialDatabaseStore(config)
    progress = queue.Queue()
    result = _subprocess_run_trial(0, config.model_dump_json(), progress, database_path=store.path_for(0))
    assert isinstance(result, OSError)
    assert progress.queue[-1] == (0, "exception", None)
    assert not any(message == "done" for _, message, _ in progress.queue)
    # The failed path is not a completed descriptor, and its handle is closed.
    store.path_for(0).rename(store.directory / "failed.sqlite")
    store.close()


def test_fail_fast_waits_for_file_users(config):
    sim = pax.MultiSimulation(config)
    sim.begin_sample_callback(fail_first_trial)
    with pytest.raises(RuntimeError, match="intentional trial failure"):
        sim.run(max_processes=2)
    assert 0 not in sim.trial_databases
    assert not sim._trial_database_store._pending
    sim._trial_database_store.close()


def test_sequential_handoff_preserves_output_files(config, tmp_path):
    config.outputs.disk = tmp_path / "results.pxsim"
    result = pax.MultiSimulation(config).sequential_run()
    assert result.trial_databases
    files = list(tmp_path.glob("results*.pxsim"))
    assert len(files) == 1
    restored = pax.SimulationTables.from_file(files[0])
    assert restored.n_total_samples == result.n_total_samples
    assert restored.trial_databases == {}


def make_trial(path, trial, *, definition="same", fare_id=1, user_version=0):
    """Small fixtures expose merge policy failures without running the engine."""
    from passengersim.driver._trial_database import TrialDatabase

    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE leg_detail (trial INTEGER, sold INTEGER);"
        "CREATE TABLE path_defs (path_id INTEGER PRIMARY KEY, name TEXT);"
        "CREATE TABLE fare_defs (fare_id INTEGER PRIMARY KEY, price REAL);"
        "CREATE TABLE fare_detail (trial INTEGER, fare_id INTEGER, sold INTEGER);"
    )
    connection.execute("INSERT INTO leg_detail VALUES (?, 5)", (trial,))
    connection.execute("INSERT INTO path_defs VALUES (1, ?)", (definition,))
    connection.execute("INSERT INTO fare_defs VALUES (?, 10)", (fare_id,))
    connection.execute("INSERT INTO fare_detail VALUES (?, ?, 7)", (trial, fare_id))
    connection.execute(f"PRAGMA user_version={user_version}")
    schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
    connection.commit()
    connection.close()
    return TrialDatabase(path, trial, schema_version, user_version, sqlite3.sqlite_version)


def test_merge_is_atomic_and_remaps_fares(config):
    store = TrialDatabaseStore(config)
    first = make_trial(store.path_for(1), 1, fare_id=100)
    second = make_trial(store.path_for(0), 0, definition="conflict", fare_id=200)
    store.merge(first)  # Intentionally out of order.
    with pytest.raises(ValueError, match="Conflicting static definitions in path_defs"):
        store.merge(second)
    assert store.merged == {1}
    assert second.path.exists()
    with sqlite3.connect(store.consolidated_path) as connection:
        assert connection.execute("SELECT * FROM leg_detail").fetchall() == [(1, 5)]
        assert connection.execute("SELECT * FROM fare_detail").fetchall() == [(1, 100, 7)]
    connection.close()
    with sqlite3.connect(second.path) as connection:
        connection.execute("UPDATE path_defs SET name='same'")
    connection.close()
    store.merge(second)
    with pytest.raises(ValueError, match="already been merged"):
        store.merge(second)
    with sqlite3.connect(store.consolidated_path) as connection:
        assert connection.execute("SELECT * FROM fare_detail ORDER BY trial").fetchall() == [(0, 100, 7), (1, 100, 7)]
        assert connection.execute("SELECT count(*) FROM fare_defs").fetchone() == (1,)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    connection.close()
    store.close()


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("CREATE TABLE custom (x)", "No merge policy"),
        ("CREATE VIEW custom AS SELECT * FROM leg_detail", "No merge policy"),
        ("UPDATE leg_detail SET trial=99", "Unexpected trial"),
        ("ALTER TABLE leg_detail ADD COLUMN extra", "schema differs"),
        ("DROP TABLE leg_detail", "table sets differ"),
        ("CREATE INDEX changed ON leg_detail (trial)", "indexes differ"),
        ("UPDATE fare_detail SET fare_id=99", "Unknown fare_id"),
        ("PRAGMA user_version=1", "user_version changed"),
    ],
)
def test_rejected_trial_leaves_parent_unchanged(config, mutation, message):
    store = TrialDatabaseStore(config)
    store.merge(make_trial(store.path_for(0), 0))
    second = make_trial(store.path_for(1), 1)
    with sqlite3.connect(second.path) as connection:
        connection.execute(mutation)
    connection.close()
    with pytest.raises(ValueError, match=message):
        store.merge(second)
    with sqlite3.connect(store.consolidated_path) as connection:
        assert connection.execute("SELECT * FROM leg_detail").fetchall() == [(0, 5)]
    connection.close()
    assert second.path.exists()
    store.close()


def test_first_merge_failure_rolls_back_schema(config):
    store = TrialDatabaseStore(config)
    trial = make_trial(store.path_for(0), 0)
    with sqlite3.connect(trial.path) as connection:
        connection.execute("UPDATE leg_detail SET trial=9")
    connection.close()
    with pytest.raises(ValueError, match="Unexpected trial"):
        store.merge(trial)
    with sqlite3.connect(store.consolidated_path) as connection:
        assert connection.execute("SELECT name FROM sqlite_master").fetchall() == []
    connection.close()
    assert trial.path.exists()
    store.close()


def test_bookings_key_includes_trial():
    from passengersim.database.tables import create_table_bookings_by_timeframe

    connection = sqlite3.connect(":memory:")
    create_table_bookings_by_timeframe(connection, primary_key=True)
    for trial in (0, 1):
        connection.execute(
            "INSERT INTO bookings_by_timeframe (scenario,trial,carrier,booking_class,days_prior) "
            "VALUES ('test',?,'AL1','Y',0)",
            (trial,),
        )
    assert connection.execute("SELECT count(*) FROM bookings_by_timeframe").fetchone() == (2,)
    connection.close()


def test_sequential_and_parallel_database_reports_match(config):
    import pandas as pd

    from passengersim.database.common_queries import bid_price_history

    sequential = pax.MultiSimulation(config).sequential_run()
    parallel = pax.MultiSimulation(config).run(max_processes=2)
    reference_config = config.model_copy(deep=True)
    reference_config.db.filename = config.db.result_directory / "reference.sqlite"
    reference_config.db.result_directory = None
    reference_config.db.result_storage = "persistent"
    reference_sim = pax.Simulation(reference_config)
    reference = reference_sim.run()
    # Fare IDs are process-local, so compare their definitions rather than IDs.
    for result in (sequential, parallel):
        assert result.database_status["complete"]
        assert result.db.execute("SELECT STDEV(sold) FROM leg_detail").fetchone()[0] is not None
    for query in (
        "SELECT trial, sample, leg_id, sold, revenue FROM leg_detail ORDER BY trial,sample,leg_id",
        "SELECT d.trial, d.sample, f.carrier, f.orig, f.dest, f.booking_class, f.price, d.sold "
        "FROM fare_detail d JOIN fare_defs f USING (fare_id) "
        "ORDER BY d.trial,d.sample,f.carrier,f.orig,f.dest,f.booking_class,f.price",
        "SELECT trial,carrier,booking_class,days_prior,tot_sold FROM bookings_by_timeframe "
        "ORDER BY trial,carrier,booking_class,days_prior",
    ):
        assert sequential.db.execute(query).fetchall() == parallel.db.execute(query).fetchall()
        assert sequential.db.execute(query).fetchall() == reference.db.execute(query).fetchall()
    pd.testing.assert_frame_equal(
        bid_price_history(sequential.db, scenario=config.scenario, burn_samples=1),
        bid_price_history(parallel.db, scenario=config.scenario, burn_samples=1),
    )
    parallel.db.execute("CREATE TABLE user_notes (note TEXT)")
    parallel.db.execute("INSERT INTO user_notes VALUES ('after workers exit')")
    parallel.db.connection.commit()
    assert parallel.db.execute("SELECT note FROM user_notes").fetchone() == ("after workers exit",)


def create_stale_report_cache(sim):
    sim.db.execute("CREATE TABLE IF NOT EXISTS bid_price_general_summary (stale INTEGER)")


def test_worker_report_caches_are_recomputed(config):
    config.db.write_items.add("leg")
    from passengersim.database.common_queries import bid_price_history

    sim = pax.MultiSimulation(config)
    sim.end_sample_callback(create_stale_report_cache)
    summary = sim.run(max_processes=2)
    assert summary.db.execute("SELECT name FROM sqlite_master WHERE name LIKE 'bid_price%summary'").fetchall() == []
    frame = bid_price_history(summary.db, scenario=config.scenario, burn_samples=1)
    assert not frame.empty
    assert "bid_price_mean" in frame.columns
    assert summary.db.execute("SELECT count(*) FROM bid_price_general_summary").fetchone()[0] > 0
    assert summary.db.execute("SELECT count(DISTINCT trial) FROM leg_detail").fetchone() == (2,)


def fail_second_trial(sim):
    if sim.eng.trial == 1:
        raise RuntimeError("intentional second trial failure")


@pytest.mark.parametrize("fail_fast", [True, False])
def test_failure_retains_consolidated_success(config, fail_fast):
    sim = pax.MultiSimulation(config)
    sim.begin_sample_callback(fail_second_trial)
    if fail_fast:
        with pytest.raises(RuntimeError, match="second trial failure"):
            sim.sequential_run()
    else:
        summary = sim.sequential_run(fail_fast=False)
        assert summary.db is sim.db
    assert sim.database_status["completed"] == [0]
    assert set(sim.database_status["failed"]) == {1}
    assert not sim.database_status["complete"]
    assert sim.db.execute("SELECT DISTINCT trial FROM leg_detail").fetchall() == [(0,)]


def test_cancelled_async_run_retains_finished_workers(config):
    from passengersim.mp_executor import JobExecutor

    sim = pax.MultiSimulation(config)
    with JobExecutor(max_workers=1) as jobber:
        future = sim._run_asynchronously(jobber=jobber)
        assert future.cancel()
        future._monitor_thread.join(timeout=30)
        assert not future._monitor_thread.is_alive()
    status = sim.database_status
    assert set(status["completed"]) | set(status["cancelled"]) == {0, 1}
    if status["completed"]:
        assert sim.db.execute("SELECT DISTINCT trial FROM leg_detail ORDER BY trial").fetchall() == [
            (trial,) for trial in status["completed"]
        ]


def test_rerun_does_not_close_earlier_result(config):
    sim = pax.MultiSimulation(config)
    first = sim.sequential_run()
    first_db = first.db
    second = sim.sequential_run()
    assert first.db is first_db
    assert second.db is sim.db
    assert first.db.filename != second.db.filename
    assert first.db.execute("SELECT count(DISTINCT trial) FROM leg_detail").fetchone() == (2,)


def test_disk_full_rolls_back_trial(config, monkeypatch):
    import passengersim.driver._trial_database as module

    store = TrialDatabaseStore(config)
    store.merge(make_trial(store.path_for(0), 0))
    second = make_trial(store.path_for(1), 1)
    with sqlite3.connect(second.path) as connection:
        connection.executemany("INSERT INTO leg_detail VALUES (1, ?)", [("x" * 10000,)] * 100)
    connection.close()
    connect = sqlite3.connect

    def limited_connect(path, **kwargs):
        connection = connect(path, **kwargs)
        pages = connection.execute("PRAGMA page_count").fetchone()[0]
        connection.execute(f"PRAGMA max_page_count={pages}")
        return connection

    with monkeypatch.context() as patch:
        patch.setattr(module.sqlite3, "connect", limited_connect)
        with pytest.raises(sqlite3.OperationalError, match="full"):
            store.merge(second)
    assert store.merged == {0} and second.path.exists()
    with sqlite3.connect(store.consolidated_path) as connection:
        assert connection.execute("SELECT * FROM leg_detail").fetchall() == [(0, 5)]
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    connection.close()
    store.close()


def test_cleanup_failure_does_not_undo_committed_merge(config, monkeypatch, caplog):
    from pathlib import Path

    store = TrialDatabaseStore(config)
    trial = make_trial(store.path_for(0), 0)
    unlink = Path.unlink

    def blocked(path, *args, **kwargs):
        if path == trial.path:
            raise PermissionError("test file held open")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", blocked)
        store.merge(trial)
    assert "could not remove" in caplog.text
    assert store.merged == {0} and trial.path.exists()
    assert store.completed[0].path == store.consolidated_path
    store.close()


def test_config_policy_ignores_only_paths_times_and_set_order(config):
    import json

    store = TrialDatabaseStore(config)
    for trial in (0, 1):
        descriptor = make_trial(store.path_for(trial), trial)
        with sqlite3.connect(descriptor.path) as connection:
            connection.execute(
                "CREATE TABLE runtime_configs (scenario TEXT, pxsim_version TEXT, configs TEXT, updated_at TEXT)"
            )
            cfg = {
                "db": {"filename": str(descriptor.path), "write_items": ["fare", "leg"][:: 1 if trial else -1]},
                "outputs": {"reports": ["one", "two"][:: 1 if trial else -1]},
                "scenario": "same",
            }
            connection.execute("INSERT INTO runtime_configs VALUES ('same','test',?,?)", (json.dumps(cfg), str(trial)))
        connection.close()
        store.merge(descriptor)
    assert store.merged == {0, 1}
    store.close()


def add_unknown_table_on_second_trial(sim):
    if sim.eng.trial == 1:
        sim.db.execute("CREATE TABLE IF NOT EXISTS custom_data (value)")


def test_merge_failure_excludes_trial_from_summary(config):
    sim = pax.MultiSimulation(config)
    sim.begin_sample_callback(add_unknown_table_on_second_trial)
    summary = sim.run(max_processes=2, fail_fast=False)
    assert summary.n_total_samples == 2  # One successful trial, burn excluded.
    assert summary.database_status["completed"] == [0]
    assert "No merge policy" in summary.database_status["failed"][1]
    assert not summary.database_status["complete"]
    assert set(summary.trial_databases) == {0}
    assert summary.db.execute("SELECT DISTINCT trial FROM leg_detail").fetchall() == [(0,)]


def exit_first_trial(sim):
    if sim.eng.trial == 0:
        raise SystemExit("intentional worker exit")


def test_worker_base_exception_does_not_hang_monitor(config):
    from passengersim.mp_executor import JobExecutor

    sim = pax.MultiSimulation(config)
    sim.begin_sample_callback(exit_first_trial)
    with JobExecutor(max_workers=2) as jobber:
        future = sim._run_asynchronously(jobber=jobber, fail_fast=False)
        result = future.result(timeout=30)
        future._monitor_thread.join(timeout=30)
    assert result.database_status["completed"] == [1]
    assert result.database_status["failed"] == {0: "intentional worker exit"}
    assert result.db is sim.db


def test_submission_failure_retains_already_completed_trial(config):
    from types import SimpleNamespace

    class FailingExecutor:
        def __init__(self):
            self.submissions = 0

        def submit(self, function, *args):
            self.submissions += 1
            if self.submissions == 2:
                raise RuntimeError("intentional submission failure")
            future = concurrent.futures.Future()
            future.set_result(function(*args))
            return future

    sim = pax.MultiSimulation(config)
    with pytest.raises(RuntimeError, match="submission failure"):
        sim._run_asynchronously(jobber=SimpleNamespace(executor=FailingExecutor()))
    assert sim.database_status == {"completed": [0], "failed": {}, "cancelled": [1], "complete": False}
    assert sim.db.execute("SELECT DISTINCT trial FROM leg_detail").fetchall() == [(0,)]


def test_parent_connection_keeps_sqlite_thread_restriction(config):
    summary = pax.MultiSimulation(config).run(max_processes=2)
    database = summary.db
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(database.execute, "SELECT count(*) FROM leg_detail")
        with pytest.raises(sqlite3.ProgrammingError, match="thread"):
            future.result(timeout=5)
    assert database.execute("SELECT count(DISTINCT trial) FROM leg_detail").fetchone() == (2,)


@pytest.mark.parametrize("persistent", [False, True])
def test_shared_close_invalidates_all_aliases(config, persistent):
    config.db.result_storage = "persistent" if persistent else "temporary"
    sim = pax.MultiSimulation(config)
    summary = sim.sequential_run()
    database = summary.db
    raw = database.connection
    path = sim._trial_database_store.directory
    summary.close()
    sim.close()
    database.close()
    assert summary.db is None and sim.db is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        raw.execute("SELECT 1")
    assert path.exists() == persistent


def test_context_manager_and_raw_connection_keep_owner(config):
    sim = pax.MultiSimulation(config)
    summary = sim.sequential_run()
    raw = summary.db.connection
    directory = sim._trial_database_store.directory
    del sim, summary
    gc.collect()
    assert directory.exists()
    assert raw.execute("SELECT count(DISTINCT trial) FROM leg_detail").fetchone() == (2,)
    del raw
    gc.collect()
    assert not directory.exists()
    with pax.MultiSimulation(config) as sim:
        summary = sim.sequential_run()
    assert summary.db is None


def test_close_failure_can_be_retried(config, monkeypatch):
    summary = pax.MultiSimulation(config).sequential_run()
    store = summary._trial_database_store
    with monkeypatch.context() as patch:

        def blocked():
            raise PermissionError("external file handle")

        patch.setattr(store._temporary, "cleanup", blocked)
        with pytest.raises(PermissionError):
            summary.close()
    assert summary.db is None
    assert store.directory.exists()
    summary.close()
    assert not store.directory.exists()


def test_parent_initializer_runs_on_caller_thread(config):
    import threading

    calls = []

    def initialize(connection):
        calls.append(threading.get_ident())
        connection.create_function("user_double", 1, lambda value: 2 * value)

    sim = pax.MultiSimulation(config, database_initializer=initialize)
    summary = sim.run(max_processes=2)
    assert calls == [threading.get_ident()]
    assert summary.db.execute("SELECT user_double(3)").fetchone() == (6,)
    assert summary.db is sim.db
    assert len(calls) == 1
    summary.close()


def test_fork_rejects_inherited_result_without_removing_parent_files(config):
    import os

    if not hasattr(os, "fork"):
        return  # Windows has no fork; spawned workers are covered separately.
    summary = pax.MultiSimulation(config).sequential_run()
    raw = summary.db.connection
    cursor = raw.execute("SELECT 1")
    path = summary._trial_database_store.directory
    pid = os.fork()
    if pid == 0:
        try:
            for operation in (
                lambda: summary.db,
                summary.close,
                lambda: raw.execute("SELECT 1"),
                lambda: cursor.fetchone(),
            ):
                try:
                    operation()
                except RuntimeError as error:
                    if "inherited process" not in str(error):
                        os._exit(2)
                else:
                    os._exit(3)
            os._exit(0)
        except BaseException:
            os._exit(4)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert path.exists() and raw.execute("SELECT 1").fetchone() == (1,)
    summary.close()


def test_raw_close_invalidates_aliases_and_is_idempotent(config):
    sim = pax.MultiSimulation(config)
    result = sim.sequential_run()
    raw = result.db.connection
    directory = sim._trial_database_store.directory
    raw.close()
    raw.close()
    result.close()
    assert result.db is None and sim.db is None
    assert not directory.exists()


@pytest.mark.parametrize("persistent", [False, True])
def test_gc_closes_connection_before_storage_cleanup_without_commit(config, persistent):
    config.db.result_storage = "persistent" if persistent else "temporary"
    observations = []

    def make_garbage_cycle():
        summary = pax.MultiSimulation(config).sequential_run()
        store = summary._trial_database_store
        raw = summary.db.connection
        raw.execute("CREATE TABLE uncommitted_probe (value INTEGER)")
        raw.commit()
        raw.execute("INSERT INTO uncommitted_probe VALUES (1)")
        if store._temporary is not None:
            cleanup = store._temporary.cleanup

            def check_closed_before_cleanup():
                # Retain a raw handle in this garbage cycle: dropping the database
                # wrapper's reference alone cannot close it before file deletion.
                try:
                    sqlite3.Connection.execute(raw, "SELECT 1")
                except sqlite3.ProgrammingError as exc:
                    observations.append("closed" if "closed" in str(exc) else str(exc))
                else:
                    observations.append("still open")
                cleanup()

            store._temporary.cleanup = check_closed_before_cleanup
        return store.directory, weakref.ref(store)

    directory, owner = make_garbage_cycle()
    gc.collect()
    assert owner() is None
    if persistent:
        with sqlite3.connect(directory / "results.sqlite") as connection:
            assert connection.execute("SELECT count(*) FROM uncommitted_probe").fetchone() == (0,)
        connection.close()
    else:
        assert observations == ["closed"]
        assert not directory.exists()


@pytest.mark.parametrize("stage", ["config", "rm_systems", "pickle", "manager", "queue"])
@pytest.mark.parametrize("persistent", [False, True])
def test_async_setup_failure_does_not_reserve_storage(
    config: pax.Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, persistent: bool
) -> None:
    """Setup errors leave no active owner, scratch directory, or reserved filename."""
    from passengersim.driver import _multiproc

    config.db.result_storage = "persistent" if persistent else "temporary"
    config.db.result_directory = tmp_path / "results"
    if persistent:
        config.db.result_directory = None
        config.db.filename = tmp_path / "exact.sqlite"
    sim = pax.MultiSimulation(config)
    shutdowns: list[bool] = []

    def fail(*args: object, **kwargs: object) -> Never:
        raise RuntimeError("intentional setup failure")

    targets = {
        "config": (sim, "_dump_config"),
        "rm_systems": (_multiproc, "export_registered_rm_systems"),
        "pickle": (_multiproc.dill, "dumps"),
        "manager": (_multiproc.multiprocessing, "Manager"),
    }
    if stage == "queue":
        manager = SimpleNamespace(Queue=fail, shutdown=lambda: shutdowns.append(True))
        monkeypatch.setattr(_multiproc.multiprocessing, "Manager", lambda: manager)
    else:
        target, attribute = targets[stage]
        monkeypatch.setattr(target, attribute, fail)
    with pytest.raises(RuntimeError, match="intentional setup failure"):
        sim._run_asynchronously(jobber=SimpleNamespace())
    sim.close()
    assert not (tmp_path / "exact.sqlite").exists()
    assert not list(tmp_path.rglob("passengersim-db-*"))
    if stage == "queue":
        assert shutdowns == [True]
    # The same exact destination remains available for the next attempt.
    store = TrialDatabaseStore(config)
    store.close()
