"""Exercise SQLite ownership and cleanup on all platforms."""

import gc
import sqlite3

import pytest

from passengersim.database import Database


def test_in_memory_database_closes_on_owner_cleanup():
    database = Database(filename=":memory:")
    connection = database.connection
    assert connection.execute("PRAGMA database_list").fetchone()[2] == ""
    del database
    gc.collect()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_borrowed_connection_survives_wrapper_cleanup():
    connection = sqlite3.connect(":memory:")
    database = Database(connection=connection)
    del database
    gc.collect()
    assert connection.execute("SELECT 1").fetchone() == (1,)
    connection.close()
