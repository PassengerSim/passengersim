"""Single-process simulations preserve Python-owned SQLite connections."""

import sqlite3

import pytest

import passengersim as pax
from passengersim.database import Database

pytestmark = [
    pytest.mark.filterwarnings("error:.*cnx.*:DeprecationWarning"),
    pytest.mark.filterwarnings("error:Binding .* is a named parameter:DeprecationWarning"),
]


def config(filename):
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 3
    cfg.simulation_controls.burn_samples = 1
    cfg.simulation_controls.show_progress_bar = False
    cfg.db = pax.config.DatabaseConfig(filename=filename)
    cfg.db.write_items = {"leg_final", "fare_final", "demand_final", "bookings"}
    cfg.outputs.disk = False
    cfg.outputs.html.filename = False
    cfg.outputs.pickle = False
    return cfg


@pytest.mark.parametrize("memory", [True, False])
def test_simulation_keeps_connection(memory, tmp_path):
    cfg = config(":memory:" if memory else tmp_path / "simulation.sqlite")
    sim = pax.Simulation(cfg)
    connection = sim.db.connection
    connection.execute("CREATE TEMP TABLE user_data(value)")
    connection.execute("INSERT INTO user_data VALUES (123)")
    summary = sim.run()
    assert sim.db.connection is connection
    assert summary.db is sim.db
    assert sim.db._db_writer is None
    assert connection.execute("SELECT * FROM user_data").fetchall() == [(123,)]
    assert connection.execute("SELECT count(*) FROM leg_detail").fetchone()[0] > 0
    assert connection.execute(
        "SELECT count(*) FROM (SELECT scenario, iteration, trial, sample, days_prior, leg_id "
        "FROM leg_detail GROUP BY 1,2,3,4,5,6 HAVING count(*) > 1)"
    ).fetchone() == (0,)
    sim.db.close()


def test_user_connection_and_transaction_survive_simulation():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE user_data(value)")
    connection.execute("INSERT INTO user_data VALUES (123)")
    sim = pax.Simulation(config(None), connection=connection)
    sim.run()
    assert sim.db.connection is connection
    assert connection.in_transaction
    sim.db.close()
    assert connection.execute("SELECT * FROM user_data").fetchall() == [(123,)]
    connection.rollback()
    assert connection.execute("SELECT * FROM user_data").fetchall() == []
    connection.close()


def test_borrowed_database_context_preserves_outer_transaction():
    connection = sqlite3.connect(":memory:")
    database = Database(connection=connection)
    with database:
        connection.execute("CREATE TABLE user_data(value)")
        connection.execute("INSERT INTO user_data VALUES (1)")
        with pytest.raises(ValueError), database:
            connection.execute("INSERT INTO user_data VALUES (2)")
            raise ValueError("rollback inner scope")
    assert connection.execute("SELECT * FROM user_data").fetchall() == [(1,)]
    assert connection.in_transaction
    database.close()
    connection.rollback()
    connection.close()


def test_backup_preserves_borrowed_connections_and_transactions():
    source = sqlite3.connect(":memory:")
    target = sqlite3.connect(":memory:")
    database = Database(connection=source)
    with pytest.raises(ValueError, match="caller-owned"):
        database.backup(target, show_progress=False)
    assert source.in_transaction
    source.commit()
    database.backup(target, show_progress=False)
    assert not source.in_transaction
    assert target.execute("SELECT count(*) FROM leg_detail").fetchone() == (0,)
    target.execute("INSERT INTO runtime_configs(scenario) VALUES ('pending')")
    with pytest.raises(ValueError, match="destination"):
        database.backup(target, show_progress=False)
    assert target.in_transaction
    database.close()
    source.close()
    target.close()


@pytest.mark.parametrize("nested", [False, True])
def test_context_preserves_whole_transaction_rollback(nested: bool) -> None:
    """SQLite's automatic rollback must not turn a constraint error into a savepoint error."""
    connection = sqlite3.connect(":memory:")
    database = Database(connection=connection)
    connection.execute("CREATE TABLE unique_values(value UNIQUE)")
    connection.commit()
    try:
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            with database:
                connection.execute("INSERT INTO unique_values VALUES (1)")
                if nested:
                    with database:
                        connection.execute("INSERT OR ROLLBACK INTO unique_values VALUES (1)")
                else:
                    connection.execute("INSERT OR ROLLBACK INTO unique_values VALUES (1)")
        assert database._contexts == []
        assert connection.execute("SELECT * FROM unique_values").fetchall() == []
        with database:
            connection.execute("INSERT INTO unique_values VALUES (2)")
        assert connection.execute("SELECT * FROM unique_values").fetchall() == [(2,)]
    finally:
        database.close()
        connection.close()


def test_context_cleanup_failure_preserves_original() -> None:
    """Attach a cleanup failure to the original exception instead of replacing it."""
    connection = sqlite3.connect(":memory:")
    database = Database(connection=connection)
    try:
        with pytest.raises(ValueError, match="original failure") as caught:
            with database:
                connection.execute("RELEASE passengersim_context_0")
                raise ValueError("original failure")
        assert "cleanup also failed" in caught.value.__notes__[0]
        with pytest.raises(sqlite3.OperationalError, match="no such savepoint"):
            with database:
                connection.execute("RELEASE passengersim_context_0")
    finally:
        database.close()
        connection.close()
