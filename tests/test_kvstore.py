"""Dual-layout mapping behavior, copying, and mutation safety."""

from __future__ import annotations

import os
import pathlib
import pickle
import sqlite3
from pathlib import Path
from typing import Any, Never

import pandas as pd
import pytest

from passengersim.utils import kvstore
from passengersim.utils.kvstore import KVStore
from passengersim.utils.pxsim_storage import IntegrityError, PayloadWriteError, StorageError


@pytest.mark.parametrize("version", [1, 2])
def test_mapping(tmp_path: Path, version: int) -> None:
    """Round-trip rich pandas values and exercise both mapping layouts."""
    store = KVStore(tmp_path / "data.pxsim", format_version=version, chunk_bytes=1024)
    frame = pd.DataFrame({"category": pd.Categorical(["a", "b"]), "number": pd.array([1, None], dtype="Int64")})
    frame.index = pd.MultiIndex.from_tuples([(1, "x"), (2, "y")], names=["one", "two"])
    store["frame"] = frame
    pd.testing.assert_frame_equal(store["frame"], frame)
    assert list(store) == ["frame"] and len(store) == 1 and "frame" in store
    assert store.get_sizes()["frame"] > 0
    store["frame"] = None
    assert store["frame"] is None
    del store["frame"]
    with pytest.raises(KeyError):
        del store["frame"]
    assert len(store) == 0
    store.validate(full=True)
    store.close()


def test_low_limit_and_membership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stream values above the SQLite limit and query metadata without unpickling."""
    store = KVStore(tmp_path / "large", format_version=2, chunk_bytes=1024)
    store.conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 4096)
    data = os.urandom(100000)
    monkeypatch.setattr(pickle, "dumps", lambda *a, **k: pytest.fail("whole-object dumps"))
    store["large"] = data
    monkeypatch.setattr(pickle, "loads", lambda *a, **k: pytest.fail("whole-object loads"))
    assert store["large"] == data
    monkeypatch.setattr(kvstore, "_ArchiveUnpickler", lambda *a, **k: pytest.fail("metadata decoded"))
    assert "large" in store and store.get_sizes()["large"] > 4096
    assert max(r[0] for r in store.conn.execute("SELECT length(data) FROM pxsim_chunks")) < 4096
    store.close()


@pytest.mark.parametrize("version", [1, 2])
def test_copy_without_unpickling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: int) -> None:
    """Copy either source layout to version 2 without reconstructing objects."""
    source = KVStore(tmp_path / "source", format_version=version, chunk_bytes=128)
    data = os.urandom(10000)
    source["bytes"] = data
    dest = KVStore(tmp_path / "dest", format_version=2, chunk_bytes=128)
    with monkeypatch.context() as m:
        m.setattr(pickle, "loads", lambda *a, **k: pytest.fail("loads"))
        m.setattr(kvstore, "_ArchiveUnpickler", lambda *a, **k: pytest.fail("Unpickler"))
        dest.copy_entry_from(source, "bytes")
    assert dest["bytes"] == data
    dest.validate(full=True)
    source.close()
    dest.close()


class Unpicklable:
    def __reduce__(self) -> Never:
        """Inject a serialization failure before publication."""
        raise RuntimeError("serializer failed")


def test_atomic_replacement_and_outer_rollback(tmp_path: Path) -> None:
    """Preserve original entries after serialization and outer transaction failures."""
    store = KVStore(tmp_path / "data", format_version=2)
    store["a"] = 42
    with pytest.raises(PayloadWriteError):
        store["a"] = Unpicklable()
    assert store["a"] == 42
    with pytest.raises(RuntimeError), store.transaction():
        store["a"] = 11
        store["b"] = 12
        raise RuntimeError("cancel export")
    assert store["a"] == 42 and "b" not in store
    assert store.conn.execute("SELECT count(*) FROM pxsim_payloads").fetchone() == (1,)
    store.close()


@pytest.mark.parametrize("version", [1, 2])
def test_read_only_no_mutation(tmp_path: Path, version: int) -> None:
    """Read both layouts without allowing mutation or changing file bytes."""
    path = tmp_path / "data"
    s = KVStore(path, format_version=version)
    s["x"] = 123
    s.close()
    before = path.read_bytes()
    s = KVStore(path, mode="r")
    assert s["x"] == 123
    with pytest.raises(PermissionError):
        s["x"] = 4
    with pytest.raises(PermissionError):
        del s["x"]
    s.close()
    assert before == path.read_bytes()


def test_unknown_schema_not_modified(tmp_path: Path) -> None:
    """Reject unrelated SQLite files without modifying their contents."""
    path = tmp_path / "unrelated"
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE unrelated(x)")
    c.close()
    before = path.read_bytes()
    with pytest.raises(StorageError):
        KVStore(path)
    assert path.read_bytes() == before


def test_full_validation_does_not_unpickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Detect digest corruption without executing the stored pickle."""
    s = KVStore(tmp_path / "data", format_version=2)
    s["value"] = "hello"
    monkeypatch.setattr(kvstore, "_ArchiveUnpickler", lambda *a, **k: pytest.fail("unpickle"))
    s.validate(full=True)
    s.conn.execute("UPDATE pxsim_payloads SET sha256=zeroblob(32)")
    with pytest.raises(IntegrityError):
        s.validate(full=True)
    s.close()


def test_validator_rejects_orphan_payload(tmp_path: Path) -> None:
    """Reject payloads that have no owning entry."""
    store = KVStore(tmp_path / "data")
    store["x"] = 1
    store.conn.execute("DELETE FROM pxsim_entries")
    with pytest.raises(IntegrityError, match="Unreferenced"):
        store.validate()
    store.close()


@pytest.mark.parametrize(
    "schema",
    [
        "CREATE TABLE pxsim_entries(key, payload_id)",
        "CREATE TABLE kvstore(key TEXT PRIMARY KEY,value BLOB); CREATE TABLE pxsim_format(id,major,minor)",
    ],
)
def test_partial_or_mixed_schema(tmp_path: Path, schema: str) -> None:
    """Reject incomplete and mixed schemas without rewriting the source."""
    path = tmp_path / "bad"
    conn = sqlite3.connect(path)
    conn.executescript(schema)
    conn.close()
    before = path.read_bytes()
    with pytest.raises(StorageError):
        KVStore(path)
    assert path.read_bytes() == before


def test_version_and_codec_rejected(tmp_path: Path) -> None:
    """Reject future codecs and format versions with actionable errors."""
    from passengersim.utils.pxsim_storage import UnsupportedFormatError

    path = tmp_path / "data"
    store = KVStore(path)
    store["x"] = 123
    store.conn.execute("UPDATE pxsim_payloads SET codec='future'")
    store.conn.commit()
    with pytest.raises(UnsupportedFormatError):
        store["x"]
    store.conn.execute("UPDATE pxsim_format SET minor=99")
    store.conn.commit()
    store.close()
    with pytest.raises(UnsupportedFormatError):
        KVStore(path)


def test_arrays_datetimes_and_shared_references(tmp_path: Path) -> None:
    """Preserve array layouts, pandas types, shared references, and cycles."""
    import numpy as np

    store = KVStore(tmp_path / "types", chunk_bytes=128)
    base = np.arange(120).reshape(10, 12)
    values = [
        base,
        np.asfortranarray(base),
        base[::2, ::3],
        pd.DataFrame({"when": pd.date_range("2020-01-01", periods=2, tz="UTC")}),
        pd.DataFrame(),
        pd.Series([], dtype="float64"),
    ]
    for i, value in enumerate(values):
        store[str(i)] = value
        actual = store[str(i)]
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual, value)
        elif isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(actual, value)
        else:
            pd.testing.assert_series_equal(actual, value)
    shared = []
    shared.append(shared)
    store["cycle"] = [shared, shared]
    actual = store["cycle"]
    assert actual[0] is actual[1] and actual[0][0] is actual[0]
    store.close()


def test_read_only_filesystem_fallback(tmp_path: Path) -> None:
    """Fall back to read-only access when file permissions prohibit writes."""
    path = tmp_path / "readonly"
    store = KVStore(path)
    store["x"] = 1
    store.close()
    before = path.read_bytes()
    path.chmod(0o444)
    try:
        store = KVStore(path, mode="a")
        assert store["x"] == 1
        if not os.access(path, os.W_OK):
            assert store.read_only
        store.close()
        assert path.read_bytes() == before
    finally:
        path.chmod(0o644)


def test_copy_rechunks_for_lower_destination_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-chunk copied bytes for a smaller limit without unpickling."""
    source = KVStore(tmp_path / "source")
    dest = KVStore(tmp_path / "dest")
    data = os.urandom(100000)
    source["data"] = data
    dest.conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 4096)
    with monkeypatch.context() as m:
        m.setattr(kvstore, "_ArchiveUnpickler", lambda *a, **k: pytest.fail("copy decoded an object"))
        dest.copy_entry_from(source, "data")
    assert dest["data"] == data
    assert dest.conn.execute("SELECT max(length(data)) FROM pxsim_chunks").fetchone()[0] < 4096
    source.close()
    dest.close()


def test_release_failure_rolls_back_mapping(tmp_path: Path) -> None:
    """Roll back replacement when releasing the savepoint fails."""
    store = KVStore(tmp_path / "data")
    store["value"] = "original"
    connection = store.conn

    class FailReleaseOnce:
        failed = False

        def __getattr__(self, name: str) -> Any:
            """Delegate connection attributes to the wrapped SQLite connection."""
            return getattr(connection, name)

        def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
            """Inject a savepoint release failure while delegating other statements."""
            if sql.startswith("RELEASE") and not self.failed:
                self.failed = True
                raise sqlite3.OperationalError("injected commit failure")
            return connection.execute(sql, *args)

    store.conn = FailReleaseOnce()
    with pytest.raises(PayloadWriteError, match="injected commit failure"):
        store["value"] = "replacement"
    assert store["value"] == "original"
    assert not connection.in_transaction
    store.close()


@pytest.mark.parametrize("version", [1, 2])
def test_non_string_membership(version: int) -> None:
    """Non-string objects are not keys and need not be SQLite-bindable."""
    store = KVStore(format_version=version)
    try:
        store["key"] = 1
        assert "key" in store
        assert object() not in store
        assert [] not in store
        assert None not in store
    finally:
        store.close()


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("path_class", ["PosixPath", "WindowsPath"])
def test_cross_platform_paths(tmp_path: Path, version: int, path_class: str) -> None:
    """Load and copy either platform's concrete paths without changing pathlib."""
    concrete_class = getattr(pathlib, path_class)

    class SavedPath:
        def __reduce__(self) -> tuple[type[Path], tuple[str, ...]]:
            """Emit a real concrete-path pickle without constructing a foreign path."""
            return concrete_class, ("saved", "results.pxsim")

    value = {"nested": [SavedPath()], "pure": pathlib.PureWindowsPath("C:/saved/results.pxsim")}
    if concrete_class is not type(Path()):
        with pytest.raises(NotImplementedError):
            pickle.loads(pickle.dumps(value))
    source = KVStore(tmp_path / "source.pxsim", format_version=version)
    target = KVStore(tmp_path / "target.pxsim")
    try:
        source["paths"] = value
        target.copy_entry_from(source, "paths")
        for store in (source, target):
            actual = store["paths"]
            assert type(actual["nested"][0]) is type(Path())
            assert actual["nested"][0] == Path("saved/results.pxsim")
            assert actual["pure"] == value["pure"]
            assert type(actual["pure"]) is pathlib.PureWindowsPath
        assert getattr(pathlib, path_class) is concrete_class
    finally:
        target.close()
        source.close()


@pytest.mark.parametrize("version", [1, 2])
def test_append_initializes_empty_file(tmp_path: Path, version: int) -> None:
    """Allow callers to reserve an archive filename before opening storage."""
    path = tmp_path / "reserved.pxsim"
    path.touch()
    store = KVStore(path, format_version=version)
    store["value"] = 42
    store.close()
    reopened = KVStore(path, mode="r")
    assert reopened.version == version
    assert reopened["value"] == 42
    reopened.close()


@pytest.mark.parametrize("mode", ["r", "rw"])
def test_existing_only_rejects_empty_file(tmp_path: Path, mode: str) -> None:
    """Existing-only modes must not silently initialize an invalid archive."""
    path = tmp_path / "empty.pxsim"
    path.touch()
    with pytest.raises(StorageError):
        KVStore(path, mode=mode)
    assert path.read_bytes() == b""


@pytest.mark.parametrize("version", [1, 2])
def test_serialization_error_is_not_size_advice(tmp_path: Path, version: int) -> None:
    """Preserve useful serialization context without suggesting a SQLite limit."""
    store = KVStore(tmp_path / "archive", format_version=version)
    try:
        with pytest.raises(PayloadWriteError, match="serializer failed") as caught:
            store["broken"] = Unpicklable()
        assert "broken" in str(caught.value) and str(store.path) in str(caught.value)
        assert "length limit" not in str(caught.value)
        assert "chunked storage" not in str(caught.value)
        assert isinstance(caught.value.__cause__, RuntimeError)
    finally:
        store.close()


def test_actual_size_error_reports_limit(tmp_path: Path) -> None:
    """A real legacy BLOB overflow retains the actionable chunked-export advice."""
    store = KVStore(tmp_path / "archive", format_version=1)
    try:
        store.conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1024)
        with pytest.raises(PayloadWriteError, match="SQLite length limit 1024.*chunked storage"):
            store["large"] = os.urandom(8192)
    finally:
        store.close()


def test_rw_rejects_read_only_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit writable mode must never silently downgrade its access."""
    path = tmp_path / "archive"
    KVStore(path).close()
    monkeypatch.setattr(os, "access", lambda *args: False)
    with pytest.raises(PermissionError, match="not writable"):
        KVStore(path, mode="rw")
    store = KVStore(path, mode="r")
    assert store.read_only
    store.close()


def test_rw_does_not_retry_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A SQLite writable-open failure propagates rather than reopening read-only."""
    path = tmp_path / "archive"
    KVStore(path).close()
    attempts = []

    def cannot_open(*args: object, **kwargs: object) -> Never:
        attempts.append(args[0])
        error = sqlite3.OperationalError("cannot open writable archive")
        error.sqlite_errorcode = sqlite3.SQLITE_CANTOPEN
        raise error

    monkeypatch.setattr(sqlite3, "connect", cannot_open)
    with pytest.raises(sqlite3.OperationalError, match="cannot open writable archive"):
        KVStore(path, mode="rw")
    assert len(attempts) == 1 and attempts[0].endswith("?mode=rw")
