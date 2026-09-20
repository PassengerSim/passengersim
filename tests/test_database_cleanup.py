"""Exercise the lifetime of temporary SQLite files on all platforms."""

import gc

from passengersim.database import Database


def test_temporary_database_closes_before_cleanup():
    database = Database(filename=":memory:")
    filename = database.filename
    connection = database._connection
    assert filename.exists()
    del database
    gc.collect()
    assert not filename.exists()
    # Keeping a Python reference to the connection must not keep its file open.
    import sqlite3

    import pytest

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("select 1")
