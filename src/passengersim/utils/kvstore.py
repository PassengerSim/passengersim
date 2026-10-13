"""Lazy Python-object mapping over legacy and chunked PXSIM SQLite files."""

from __future__ import annotations

import io
import os
import pickle
import sqlite3
from collections.abc import Iterator, MutableMapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Literal

import lz4.frame

from .pxsim_storage import (
    CHUNK_BYTES,
    ChunkReader,
    ChunkWriter,
    IntegrityError,
    PayloadInfo,
    PayloadWriteError,
    StorageError,
    create_schema,
    detect_format,
    payload_info,
    transaction,
)


class _ArchiveUnpickler(pickle.Unpickler):
    """Restore saved concrete paths using the consuming platform's path class."""

    def find_class(self, module: str, name: str) -> Any:
        """Resolve globals, adapting concrete pathlib classes across platforms.

        Pickles record PosixPath or WindowsPath instead of the portable Path
        factory. Mapping only those constructors keeps nested configuration
        paths usable without mutating pathlib globally. Pure paths retain
        their original flavor. This does not make untrusted pickles safe.
        """
        if module == "pathlib" and name in {"PosixPath", "WindowsPath"}:
            return Path
        return super().find_class(module, name)


class KVStore(MutableMapping[str, Any]):
    """Read existing layouts unchanged and explicitly create new archives.

    Parameters
    ----------
    db_path : str or Path, default ":memory:"
        Archive location, or ``:memory:``.
    mode : {'a', 'r', 'rw', 'w'}, default 'a'
        Open/create, read-only, existing writable, or create empty storage,
        respectively. Mode 'w' refuses to overwrite nonempty existing files.
        Only mode 'a' may fall back to read-only access for an existing file.
        Mode 'rw' requires writable access; use 'r' for explicit read-only access.
    format_version : int, default 2
        Layout used only for new storage. Existing files are never migrated.
    chunk_bytes : int, default CHUNK_BYTES
        Maximum decoded chunk size for new version 2 payloads. The active
        SQLite length limit may reduce the size further.

    Raises
    ------
    FileNotFoundError
        An existing-file mode selects a missing file.
    FileExistsError
        Mode 'w' selects a nonempty file.
    PermissionError
        Creation was requested with read-only access, or mode 'rw' lacks write permission.
    ValueError
        The mode, creation version, or existing archive layout is invalid.

    Notes
    -----
    Only deserialize trusted archives: Python pickle can execute code.
    Existing files retain their format when edited. Call close when done;
    it rolls back any uncommitted caller transaction.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        mode: Literal["a", "r", "rw", "w"] = "a",
        format_version: int = 2,
        chunk_bytes: int = CHUNK_BYTES,
    ) -> None:
        """Open or create storage using the mode documented on the class."""
        if mode not in {"a", "r", "rw", "w"}:
            raise ValueError(f"Invalid storage mode {mode!r}")
        self.path = str(db_path)
        self.chunk_bytes = chunk_bytes
        self.read_only = mode == "r"
        memory = self.path == ":memory:"
        path = Path(db_path).resolve() if not memory else None
        exists = not memory and path.exists()
        create = memory or (mode == "a" and (not exists or path.stat().st_size == 0)) or mode == "w"
        if mode == "w" and exists and path.stat().st_size:
            raise FileExistsError(path)
        if not create and not exists:
            raise FileNotFoundError(path)
        if not memory and exists and not os.access(path, os.W_OK):
            if mode == "rw":
                raise PermissionError(f"PXSIM archive is not writable: {path}")
            self.read_only = True
        if create and self.read_only:
            raise PermissionError("Cannot create a read-only archive")
        if memory:
            self.conn = sqlite3.connect(":memory:")
        else:
            access = "ro" if self.read_only else "rwc" if create else "rw"
            try:
                self.conn = sqlite3.connect(path.as_uri() + f"?mode={access}", uri=True)
            except sqlite3.OperationalError as error:
                if mode != "a" or create or self.read_only or error.sqlite_errorcode != sqlite3.SQLITE_CANTOPEN:
                    raise
                self.conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
                self.read_only = True
        try:
            self.conn.execute("PRAGMA foreign_keys=ON")
            if create:
                if format_version == 2:
                    create_schema(self.conn)
                elif format_version == 1:
                    self.conn.execute("CREATE TABLE kvstore (key TEXT PRIMARY KEY, value BLOB)")
                else:
                    raise ValueError(f"Unsupported creation format {format_version}")
            self.version = detect_format(self.conn)
        except BaseException:
            self.conn.close()
            raise

    def transaction(self) -> AbstractContextManager[None]:
        """Create a savepoint context for atomic mutations.

        Returns
        -------
        contextlib.AbstractContextManager
            Context committing standalone work on success or rolling it back on
            failure, without committing an existing caller transaction.

        Raises
        ------
        PermissionError
            The archive is read-only.
        """
        self.require_writable()
        return transaction(self.conn)

    def require_writable(self) -> None:
        """Raise PermissionError if this archive was opened read-only."""
        if self.read_only:
            raise PermissionError(f"PXSIM archive is read-only: {self.path}")

    def _payload_id(self, key: str) -> int:
        """Resolve an entry reference, raising KeyError if it is absent."""
        row = self.conn.execute("SELECT payload_id FROM pxsim_entries WHERE key=?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]

    def _replace(self, key: str, payload_id: int) -> None:
        """Publish a new payload and remove the old one inside the caller transaction."""
        old = self.conn.execute("SELECT payload_id FROM pxsim_entries WHERE key=?", (key,)).fetchone()
        self.conn.execute("INSERT OR REPLACE INTO pxsim_entries VALUES(?,?)", (key, payload_id))
        if old:
            self.conn.execute("DELETE FROM pxsim_payloads WHERE id=?", old)

    def __setitem__(self, key: str, value: Any) -> None:
        """Atomically serialize a value, wrapping storage failures in PayloadWriteError."""
        try:
            with self.transaction():
                if self.version == 1:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO kvstore VALUES(?,?)", (key, lz4.frame.compress(pickle.dumps(value)))
                    )
                else:
                    sink = ChunkWriter(self.conn, "pickle", protocol=5, chunk_bytes=self.chunk_bytes)
                    pickle.Pickler(sink, protocol=5).dump(value)
                    self._replace(key, sink.finish())
        except Exception as error:
            if isinstance(error, PermissionError):
                raise
            detail = ""
            if isinstance(error, sqlite3.Error) and getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_TOOBIG:
                limit = self.conn.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)
                detail = f"; SQLite length limit {limit}"
                if self.version == 1:
                    detail += "; export with to_file to use chunked storage"
            raise PayloadWriteError(f"Error writing PXSIM {self.path}, key {key!r}{detail}: {error}") from error

    def __getitem__(self, key: str) -> Any:
        """Deserialize a trusted entry, raising KeyError if absent or StorageError if corrupt."""
        if self.version == 1:
            row = self.conn.execute("SELECT value FROM kvstore WHERE key=?", (key,)).fetchone()
            if row is None:
                raise KeyError(key)
            return _ArchiveUnpickler(io.BytesIO(lz4.frame.decompress(row[0]))).load()
        try:
            with ChunkReader(self.conn, self._payload_id(key), "pickle") as raw, io.BufferedReader(raw) as stream:
                value = _ArchiveUnpickler(stream).load()
                # Buffered bytes have already been accounted for by the raw reader.
                raw.finish()
                return value
        except StorageError as error:
            raise type(error)(f"{self.path}, key {key!r}: {error}") from error

    def __contains__(self, key: object) -> bool:
        """Check for a string key without decoding its value."""
        if not isinstance(key, str):
            return False
        table = "kvstore" if self.version == 1 else "pxsim_entries"
        return self.conn.execute(f"SELECT 1 FROM {table} WHERE key=?", (key,)).fetchone() is not None

    def __delitem__(self, key: str) -> None:
        """Atomically remove an entry, raising KeyError if it is absent."""
        with self.transaction():
            if self.version == 1:
                if not self.conn.execute("DELETE FROM kvstore WHERE key=?", (key,)).rowcount:
                    raise KeyError(key)
            else:
                ident = self._payload_id(key)
                self.conn.execute("DELETE FROM pxsim_entries WHERE key=?", (key,))
                self.conn.execute("DELETE FROM pxsim_payloads WHERE id=?", (ident,))

    def copy_entry_from(self, source: KVStore, key: str) -> None:
        """Copy an entry, avoiding object reconstruction when the layout permits.

        Parameters
        ----------
        source : KVStore
            Open source archive containing the entry.
        key : str
            Entry to copy, replacing any destination entry atomically.

        Raises
        ------
        KeyError
            The key is absent from the source.
        PermissionError
            The destination is read-only.
        StorageError
            Source metadata or the copied stream is invalid.

        Notes
        -----
        Version 2 destinations reuse encoded chunks after structural validation,
        without recomputing their digests. A smaller destination SQLite limit
        requires verified decoding and re-chunking. Legacy destinations and legacy
        sources without rowids fall back to deserializing and reserializing objects.
        Use ``validate(full=True)`` when full digest verification is required.
        """
        if self.version != 2:
            self[key] = source[key]
            return
        with self.transaction():
            if source.version == 1:
                # The historical schema has a rowid; WITHOUT ROWID variants fall
                # back to the original decoder rather than becoming unreadable.
                try:
                    row = source.conn.execute("SELECT rowid FROM kvstore WHERE key=?", (key,)).fetchone()
                except sqlite3.OperationalError:
                    self[key] = source[key]
                    return
                if row is None:
                    raise KeyError(key)
                sink = ChunkWriter(self.conn, "pickle", chunk_bytes=self.chunk_bytes)
                with source.conn.blobopen("kvstore", "value", row[0], readonly=True) as blob:
                    with lz4.frame.open(blob, "rb") as stream:
                        first = True
                        while data := stream.read(self.chunk_bytes):
                            if first and len(data) >= 2 and data[0] == 0x80:
                                self.conn.execute(
                                    "UPDATE pxsim_payloads SET pickle_protocol=? WHERE id=?", (data[1], sink.id)
                                )
                            first = False
                            sink.write(data)
                ident = sink.finish()
            else:
                old = source._payload_id(key)
                info = payload_info(source.conn, old, "pickle")
                if info[3] * 2 + 384 > self.conn.getlimit(sqlite3.SQLITE_LIMIT_LENGTH):
                    # A destination with a smaller SQLite limit cannot reuse the
                    # encoded rows. Re-chunk bytes without constructing objects.
                    sink = ChunkWriter(self.conn, "pickle", protocol=info[2], chunk_bytes=self.chunk_bytes)
                    with ChunkReader(source.conn, old, "pickle") as reader:
                        while data := reader.read(sink.size):
                            sink.write(data)
                        reader.finish()
                    ident = sink.finish()
                else:
                    ident = self.conn.execute(
                        "INSERT INTO pxsim_payloads VALUES(NULL,?,?,?,?,?,?,?,?)",
                        info,
                    ).lastrowid
                    for row in source._encoded_chunks(old, info):
                        self.conn.execute("INSERT INTO pxsim_chunks VALUES(?,?,?,?,?)", (ident, *row))
            self._replace(key, ident)

    def _encoded_chunks(self, ident: int, info: PayloadInfo) -> Iterator[tuple[int, int, bytes, bytes]]:
        """Validate structure while copying; full validation additionally hashes raw bytes."""
        raw_total = stored_total = count = 0
        size = info[3]
        cursor = self.conn.execute(
            "SELECT ordinal,raw_bytes,sha256,CASE WHEN typeof(data)='blob' AND length(data)<=? THEN data ELSE NULL END "
            "FROM pxsim_chunks WHERE payload_id=? ORDER BY ordinal",
            (size * 2 + 128, ident),
        )
        try:
            for ordinal, raw, digest, data in cursor:
                if (
                    ordinal != count
                    or not isinstance(raw, int)
                    or not 0 < raw <= size
                    or data is None
                    or not isinstance(digest, bytes)
                    or len(digest) != 32
                ):
                    raise IntegrityError(f"Invalid payload {ident} chunk {count}")
                try:
                    frame = lz4.frame.get_frame_info(data)
                except RuntimeError as error:
                    raise IntegrityError(f"Invalid frame in payload {ident} chunk {count}") from error
                if frame["content_size"] not in (0, raw):
                    raise IntegrityError(f"Invalid frame size in payload {ident} chunk {count}")
                raw_total += raw
                stored_total += len(data)
                count += 1
                yield ordinal, raw, digest, data
            if (raw_total, stored_total, count) != info[4:7]:
                raise IntegrityError(f"Payload {ident} size/count mismatch")
        finally:
            cursor.close()

    def validate(self, *, full: bool = False) -> None:
        """Check archive structure without unpickling stored objects.

        Parameters
        ----------
        full : bool, default False
            Decode all payloads and verify version 2 chunk and aggregate hashes.
            Legacy archives have no payload digests; full validation only checks
            their compressed streams and can allocate entire legacy values.

        Raises
        ------
        StorageError
            The format, references, frames, or version 2 integrity checks fail.
        sqlite3.DatabaseError
            SQLite cannot read the archive.
        RuntimeError
            A legacy LZ4 stream cannot be decompressed.

        Notes
        -----
        This does not validate pickle semantics. It never executes stored pickle code.
        """
        if self.conn.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise IntegrityError("PXSIM SQLite integrity check failed")
        if detect_format(self.conn) == 1:
            if full:
                for key in self:
                    row = self.conn.execute("SELECT value FROM kvstore WHERE key=?", (key,)).fetchone()
                    lz4.frame.decompress(row[0])
            return
        if self.conn.execute("PRAGMA foreign_key_check").fetchone():
            raise IntegrityError("PXSIM dangling payload reference")
        owners = list(self.conn.execute("SELECT payload_id FROM pxsim_entries"))
        for (ident,) in owners:
            payload_info(self.conn, ident, "pickle")
        identifiers = set(self.conn.execute("SELECT id FROM pxsim_payloads"))
        if len(owners) != len(set(owners)) or set(owners) != identifiers:
            raise IntegrityError("Unreferenced or multiply owned PXSIM payload")
        if self.conn.execute(
            "SELECT 1 FROM pxsim_chunks LEFT JOIN pxsim_payloads ON payload_id=id WHERE id IS NULL LIMIT 1"
        ).fetchone():
            raise IntegrityError("PXSIM chunks reference a missing payload")
        for (ident,) in identifiers:
            info = payload_info(self.conn, ident)
            if full:
                with ChunkReader(self.conn, ident) as reader:
                    reader.finish()
            else:
                for _ in self._encoded_chunks(ident, info):
                    pass

    def vacuum(self) -> None:
        """Reclaim unused SQLite pages after committed deletions.

        Raises
        ------
        PermissionError
            The archive is read-only.
        sqlite3.OperationalError
            An active transaction or another SQLite condition prevents compaction.
        """
        self.require_writable()
        self.conn.execute("VACUUM")

    def close(self) -> None:
        """Close the connection, rolling back any uncommitted transaction."""
        self.conn.close()

    def get_sizes(self) -> dict[str, int]:
        """Return compressed payload sizes without loading stored values.

        Returns
        -------
        dict[str, int]
            Stored bytes by key, excluding SQLite pages and metadata overhead.
        """
        if self.version == 1:
            return dict(self.conn.execute("SELECT key,length(value) FROM kvstore"))
        return dict(
            self.conn.execute("SELECT key,stored_bytes FROM pxsim_entries JOIN pxsim_payloads ON payload_id=id")
        )

    def __len__(self) -> int:
        """Return the number of stored keys without decoding values."""
        table = "kvstore" if self.version == 1 else "pxsim_entries"
        return self.conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def __iter__(self) -> Iterator[str]:
        """Yield stored keys in unspecified order without decoding values."""
        table = "kvstore" if self.version == 1 else "pxsim_entries"
        cursor = self.conn.execute(f"SELECT key FROM {table}")
        try:
            for (key,) in cursor:
                yield key
        finally:
            cursor.close()
