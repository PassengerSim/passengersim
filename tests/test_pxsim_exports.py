"""Summary exports preserve lazy values and recover from publication failures."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Never

import pandas as pd
import pytest

import passengersim as pax
from passengersim.summaries import generic
from passengersim.utils import kvstore
from passengersim.utils.kvstore import KVStore


def save(tmp_path: Path) -> Path:
    """Create a minimal archive for export regression tests."""
    result = pax.SimulationTables(data={"unknown": pd.DataFrame({"x": [1, 2]})}, n_total_samples=3)
    return result.to_file(tmp_path / "output.pxsim", add_timestamp_ext=False)


def test_unloaded_resave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resave lazy entries without constructing their Python objects."""
    path = save(tmp_path)
    with pax.SimulationTables.from_file(path) as result:
        with monkeypatch.context() as m:
            m.setattr(kvstore, "_ArchiveUnpickler", lambda *a, **k: pytest.fail("unloaded entry decoded"))
            result.to_file(tmp_path / "copy.pxsim", add_timestamp_ext=False)
        with pax.SimulationTables.from_file(tmp_path / "copy.pxsim", lazy=False) as copied:
            pd.testing.assert_frame_equal(copied._data["unknown"], pd.DataFrame({"x": [1, 2]}))


def test_failed_same_path_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve source bytes and the read-only handle after publication failure."""
    path = save(tmp_path)
    before = path.read_bytes()
    with pax.SimulationTables.from_file(path, read_only=True) as result:
        with monkeypatch.context() as m:

            def fail(*args: Any) -> Never:
                """Inject an atomic publication failure."""
                raise OSError("replace failed")

            m.setattr(generic.os, "replace", fail)
            with pytest.raises(OSError, match="replace failed"):
                result.to_file(path, add_timestamp_ext=False)
        assert path.read_bytes() == before
        assert result._file_store.read_only
        assert result._file_store["unknown"].x.tolist() == [1, 2]
    assert not list(tmp_path.glob(".pxsim-*"))


def test_read_only_removal_atomic(tmp_path: Path) -> None:
    """Reject read-only deletion before changing the in-memory result."""
    path = save(tmp_path)
    with pax.SimulationTables.from_file(path, lazy=False, read_only=True) as result:
        with pytest.raises(PermissionError):
            result.remove_data("unknown")
        assert "unknown" in result._data and "unknown" in result._file_store


def test_modified_and_cleared_lazy_values(tmp_path: Path) -> None:
    """Preserve in-place edits and intentional clearing of lazy attributes."""
    path = save(tmp_path)
    with pax.SimulationTables.from_file(path, lazy=False) as result:
        result._data["unknown"].loc[0, "x"] = 99
        result.config = None
        result.callback_data = pax.callbacks.CallbackData()
        result.to_file(path, add_timestamp_ext=False)
    with pax.SimulationTables.from_file(path, lazy=False) as result:
        assert result._data["unknown"].loc[0, "x"] == 99
        assert result.config is None and not result.callback_data
        assert result._file_store.version == 2


def test_legacy_to_v2(tmp_path: Path) -> None:
    """Export legacy results as version 2 while preserving saved metadata."""
    source = Path(__file__).parent / "data/pxsim/legacy-summary.pxsim"
    with pax.SimulationTables.from_file(source) as result:
        path = result.to_file(tmp_path / "converted.pxsim", add_timestamp_ext=False)
    with pax.SimulationTables.from_file(path) as result:
        assert result.config is not None
        assert result.callback_data.begin_sample[0] == {"answer": 42}
        assert result._file_store.version == 2
        assert not result.database_available


def test_explicit_remove_legacy(tmp_path: Path) -> None:
    """Delete requested legacy entries without migrating the remaining file."""
    source = Path(__file__).parent / "data/pxsim/legacy-summary.pxsim"
    path = tmp_path / "legacy.pxsim"
    path.write_bytes(source.read_bytes())
    with pax.SimulationTables.from_file(path) as result:
        result.remove_data(["unknown_report", "stored_error"])
    s = KVStore(path, mode="r")
    assert s.version == 1 and "unknown_report" not in s
    s.close()


def test_archive_context_does_not_relax_new_input() -> None:
    """Keep current schedule validation strict outside archive restoration."""
    from pydantic import ValidationError

    path = Path(pax.__file__).parent / "networks/us25/baseline.pxsim"
    with pax.SimulationTables.from_file(path) as result:
        historical = result.config.model_dump()
    with pytest.raises(ValidationError, match="non-positive duration"):
        pax.Config.model_validate(historical)


def test_archived_custom_rm_does_not_register_it() -> None:
    """Restore historical RM names without changing the global RM registry."""
    from pydantic import ValidationError

    from passengersim.rm.systems import check_registered_rm_system, list_registered_rm_systems
    from passengersim.summaries.generic import _restore_file_config

    path = Path(__file__).parent / "data/pxsim/legacy-custom-rm.pxsim"
    before = set(list_registered_rm_systems())
    with pax.SimulationTables.from_file(path) as result:
        assert result.config.carriers["AL1"].rm_system == "EX"
        saved = result.config.model_dump()
    assert set(list_registered_rm_systems()) == before
    missing = "__pxsim_unregistered_archive_test__"
    saved["carriers"]["AL1"]["rm_system"] = missing
    saved["carriers"]["AL1"]["rm_system_options"] = None
    assert _restore_file_config(saved).carriers["AL1"].rm_system == missing
    assert not check_registered_rm_system(missing)
    with pytest.raises(ValidationError, match="unknown RM system " + missing):
        pax.Config.model_validate(saved)


def test_export_serialization_failure_preserves_destination(tmp_path: Path) -> None:
    """Leave the existing destination intact when serialization fails."""
    path = save(tmp_path)
    before = path.read_bytes()

    class CannotSave:
        def __reduce__(self) -> Never:
            """Inject a serialization failure before publication."""
            raise RuntimeError("cannot serialize")

    result = pax.SimulationTables(data={"bad": CannotSave()})
    with pytest.raises(ValueError, match="cannot serialize"):
        result.to_file(path, add_timestamp_ext=False)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".pxsim-*"))


def test_disk_full_export_preserves_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave the destination intact and remove staging files after disk-full errors."""
    import os

    path = save(tmp_path)
    before = path.read_bytes()
    original = KVStore.__init__

    def limited(self, *args: Any, **kwargs: Any) -> None:
        """Lower SQLite limits to exercise chunking or disk-full rollback."""
        original(self, *args, **kwargs)
        if kwargs.get("mode") == "w":
            self.conn.execute("PRAGMA max_page_count=64")

    monkeypatch.setattr(KVStore, "__init__", limited)
    result = pax.SimulationTables(data={"large": os.urandom(1024 * 1024)})
    with pytest.raises(ValueError, match="full"):
        result.to_file(path, add_timestamp_ext=False)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".pxsim-*"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits and umask")
@pytest.mark.parametrize(
    "mask,existing,expected",
    [(0o022, None, 0o644), (0o002, None, 0o664), (0o077, None, 0o600), (0o077, 0o664, 0o664), (0o000, 0o600, 0o600)],
)
def test_export_permissions(tmp_path: Path, mask: int, existing: int | None, expected: int) -> None:
    """Honor umask for new archives and retain permissions when replacing them."""
    path = tmp_path / "result.pxsim"
    if existing is not None:
        path.touch()
        path.chmod(existing)
    # Set umask only in a subprocess: it is process-wide, not thread-local.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sys; from passengersim import SimulationTables; "
            "os.umask(int(sys.argv[2])); "
            "SimulationTables(n_total_samples=3).to_file(sys.argv[1], add_timestamp_ext=False)",
            str(path),
            str(mask),
        ],
        check=True,
    )
    assert stat.S_IMODE(path.stat().st_mode) == expected
    assert not list(tmp_path.glob(".pxsim-*"))
    with pax.SimulationTables.from_file(path) as loaded:
        assert loaded.n_total_samples == 3


@pytest.mark.parametrize("version", [1, 2])
def test_load_read_only_archive_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: int) -> None:
    """Loading a shared archive still works when later edits are not permitted."""
    path = tmp_path / "readonly.pxsim"
    store = KVStore(path, format_version=version)
    store["_metadata_"] = {}
    store["_n_total_samples_"] = 3
    store["unknown"] = [1, 2]
    store.close()
    before = path.read_bytes()
    monkeypatch.setattr(os, "access", lambda *args: False)
    with pax.SimulationTables.from_file(path, read_latest=False) as loaded:
        assert loaded._file_store.read_only
        assert loaded._file_store["unknown"] == [1, 2]
    assert path.read_bytes() == before
