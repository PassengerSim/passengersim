"""Chunk streams are bounded, transactional, and integrity checked."""

from __future__ import annotations

import io
import os
import sqlite3
from collections.abc import Iterator

import pytest

from passengersim.utils.pxsim_storage import (
    ChunkReader,
    ChunkWriter,
    IntegrityError,
    create_schema,
    detect_format,
    transaction,
)


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """Provide an isolated version 2 in-memory archive."""
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    yield conn
    conn.close()


@pytest.mark.parametrize("length", [0, 1, 1023, 1024, 1025, 100000])
def test_stream(conn: sqlite3.Connection, length: int) -> None:
    """Round-trip boundary-sized payloads below a reduced per-row limit."""
    data = os.urandom(length)
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 4096)
    with transaction(conn):
        writer = ChunkWriter(conn, "pickle", chunk_bytes=1024)
        assert writer.write(data) == len(data)
        ident = writer.finish()
    with ChunkReader(conn, ident) as raw, io.BufferedReader(raw) as stream:
        assert stream.read() == data
    assert detect_format(conn) == 2
    largest = conn.execute("SELECT max(length(data)) FROM pxsim_chunks").fetchone()[0]
    assert largest is None if not length else largest < 4096


@pytest.mark.parametrize(
    "damage",
    [
        "DELETE FROM pxsim_chunks WHERE ordinal=0",
        "UPDATE pxsim_chunks SET sha256=zeroblob(32) WHERE ordinal=0",
        "UPDATE pxsim_payloads SET raw_bytes=raw_bytes+1",
        "UPDATE pxsim_chunks SET data=data || x'00' WHERE ordinal=0",
    ],
)
def test_corruption(conn: sqlite3.Connection, damage: str) -> None:
    """Reject missing chunks and damaged counts, frames, and checksums."""
    with transaction(conn):
        writer = ChunkWriter(conn, "pickle", chunk_bytes=100)
        writer.write(b"hello" * 100)
        ident = writer.finish()
    conn.execute(damage)
    with pytest.raises((IntegrityError, sqlite3.OperationalError)):
        with ChunkReader(conn, ident) as stream:
            stream.finish()


def test_nested_rollback(conn: sqlite3.Connection) -> None:
    """Roll back inner work while preserving the enclosing transaction."""
    with transaction(conn):
        with pytest.raises(RuntimeError), transaction(conn):
            writer = ChunkWriter(conn, "pickle")
            writer.write(b"bad")
            writer.finish()
            raise RuntimeError("cancel")
        assert conn.execute("SELECT count(*) FROM pxsim_payloads").fetchone() == (0,)
        writer = ChunkWriter(conn, "pickle")
        writer.write(b"good")
        ident = writer.finish()
    assert ChunkReader(conn, ident).read() == b"good"
