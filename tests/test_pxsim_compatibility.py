"""Frozen pre-v2 files must stay readable without rewriting their bytes."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import passengersim as pax

FIXTURES = Path(__file__).parent / "data" / "pxsim"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


@pytest.mark.parametrize("lazy", [True, False])
@pytest.mark.parametrize("name", list(MANIFEST))
def test_historical_files(name: str, lazy: bool) -> None:
    """Load frozen archives eagerly and lazily without changing bytes or schemas."""
    path = (Path(pax.__file__).parent / "networks" / name) if name.startswith("us25/") else FIXTURES / name
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == MANIFEST[name]["sha256"]
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        schema = conn.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    with pax.SimulationTables.from_file(path, lazy=lazy, read_latest=False) as result:
        assert result.n_total_samples > 0
        assert result.config is not None
        assert result.file_info()
        if name == "legacy-summary.pxsim":
            assert result.n_total_samples == 42
            pd.testing.assert_frame_equal(
                result._file_store["unknown_report"],
                pd.DataFrame({"value": [1.0, 2.0]}, index=pd.Index(["a", "b"], name="label")),
            )
            assert isinstance(result._file_store["stored_error"], ValueError)
            assert result.callback_data.begin_sample[0] == {"answer": 42}
            assert not result.database_available
        else:
            assert not result.carriers.empty
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        assert conn.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall() == schema
