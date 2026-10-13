"""Opt-in tests crossing real SQLite BLOB and signed-32-bit size boundaries.

Run with PXSIM_LARGE_TESTS=1 and a single worker; allow about 15 GB of scratch
space. Each case releases its data before loading the saved archive.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import pytest

import passengersim as pax

pytestmark = pytest.mark.skipif(
    os.environ.get("PXSIM_LARGE_TESTS") != "1", reason="opt-in multi-gigabyte storage tests"
)


def inspect_archive(path: Path, serializer: str) -> None:
    """Ensure this case really crosses the compressed SQLite value limit."""
    with sqlite3.connect(path) as conn:
        assert (
            conn.execute("SELECT max(stored_bytes) FROM pxsim_payloads WHERE serializer=?", (serializer,)).fetchone()[0]
            > 1_000_000_000
        )
        assert conn.execute("SELECT max(length(data)) FROM pxsim_chunks").fetchone()[0] < 2 * 1024 * 1024


@pytest.mark.parametrize("size", [1_050_000_000, 2_200_000_000])
def test_large_summary(tmp_path: Path, size: int) -> None:
    """Round-trip a summary whose compressed payload exceeds the BLOB limit."""
    data = os.urandom(size)
    digest = hashlib.sha256(data).digest()
    result = pax.SimulationTables(data={"large": data}, n_total_samples=1)
    path = result.to_file(tmp_path / "large.pxsim", add_timestamp_ext=False)
    result.close()
    del result, data
    inspect_archive(path, "pickle")
    with pax.SimulationTables.from_file(path) as restored:
        value = restored._file_store["large"]
        assert len(value) == size and hashlib.sha256(value).digest() == digest
