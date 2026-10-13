"""Versioned, bounded SQLite payload streams used by PXSIM archives."""

import hashlib
import io
import itertools
import sqlite3
from collections.abc import Buffer, Iterator
from contextlib import contextmanager

import lz4.frame

type PayloadInfo = tuple[str, str, int | None, int, int, int, int, bytes]

CHUNK_BYTES = 1024 * 1024
MAX_CHUNK_BYTES = 16 * CHUNK_BYTES
CODEC = "lz4-blocks-v1"
_SAVEPOINTS = itertools.count()
SCHEMA = """
CREATE TABLE pxsim_format (id INTEGER PRIMARY KEY CHECK(id=1), major INTEGER NOT NULL,
 minor INTEGER NOT NULL);
INSERT INTO pxsim_format VALUES (1,2,0);
CREATE TABLE pxsim_payloads (
 id INTEGER PRIMARY KEY, serializer TEXT NOT NULL, codec TEXT NOT NULL, pickle_protocol INTEGER,
 chunk_bytes INTEGER NOT NULL CHECK(chunk_bytes>0), raw_bytes INTEGER NOT NULL CHECK(raw_bytes>=0),
 stored_bytes INTEGER NOT NULL CHECK(stored_bytes>=0), chunk_count INTEGER NOT NULL CHECK(chunk_count>=0),
 sha256 BLOB NOT NULL CHECK(length(sha256)=32));
CREATE TABLE pxsim_chunks (
 payload_id INTEGER NOT NULL REFERENCES pxsim_payloads(id) ON DELETE CASCADE,
 ordinal INTEGER NOT NULL CHECK(ordinal>=0), raw_bytes INTEGER NOT NULL CHECK(raw_bytes>0),
 sha256 BLOB NOT NULL CHECK(length(sha256)=32), data BLOB NOT NULL,
 PRIMARY KEY(payload_id,ordinal));
CREATE TABLE pxsim_entries (key TEXT PRIMARY KEY,
 payload_id INTEGER NOT NULL UNIQUE REFERENCES pxsim_payloads(id));
"""


class StorageError(ValueError):
    """Malformed or unsupported PXSIM storage."""


class UnsupportedFormatError(StorageError):
    """The archive needs a reader supporting a different format or codec."""


class IntegrityError(StorageError):
    """Stored bytes do not match the archive's declared structure or digests."""


class PayloadWriteError(StorageError):
    """A payload could not be stored; the enclosing mutation is rolled back."""


def create_schema(conn: sqlite3.Connection) -> None:
    """Initialize version 2 tables in a newly created archive.

    Parameters
    ----------
    conn : sqlite3.Connection
        Empty writable connection, with no caller transaction. This uses
        ``executescript``, which commits any pending transaction.

    Notes
    -----
    Never call this on an existing archive; it does not migrate files.
    """
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)


def detect_format(conn: sqlite3.Connection) -> int:
    """Identify a supported layout without writing to the archive.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open archive connection; read-only connections are supported.

    Returns
    -------
    int
        1 for legacy storage or 2 for chunked storage.

    Raises
    ------
    StorageError
        The schema is missing, malformed, or mixes layouts.
    UnsupportedFormatError
        The declared format version is unsupported.
    """
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    v2 = {name for name in tables if name.startswith("pxsim_")}
    if v2:
        required = {"pxsim_format", "pxsim_payloads", "pxsim_chunks", "pxsim_entries"}
        if not required <= tables or "kvstore" in tables:
            raise StorageError("Incomplete or mixed PXSIM version 2 schema")
        try:
            if conn.execute("SELECT id,major,minor FROM pxsim_format").fetchall() != [(1, 2, 0)]:
                raise UnsupportedFormatError("Unsupported PXSIM version; this reader supports 2.0")
            columns = {
                "pxsim_payloads": (
                    "id serializer codec pickle_protocol chunk_bytes raw_bytes stored_bytes chunk_count sha256"
                ),
                "pxsim_chunks": "payload_id ordinal raw_bytes sha256 data",
                "pxsim_entries": "key payload_id",
            }
            for table, names in columns.items():
                conn.execute(f"SELECT {','.join(names.split())} FROM {table} LIMIT 0")
        except sqlite3.DatabaseError as error:
            raise StorageError("Malformed PXSIM version 2 schema") from error
        return 2
    if "kvstore" in tables:
        if not {"key", "value"} <= {r[1] for r in conn.execute("PRAGMA table_info(kvstore)")}:
            raise StorageError("Malformed legacy PXSIM schema")
        return 1
    raise StorageError("Not a recognized PXSIM archive")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Isolate mutations in a savepoint and roll them back on failure.

    Parameters
    ----------
    conn : sqlite3.Connection
        Writable connection whose existing transaction remains caller-owned.

    Yields
    ------
    None
        Control inside the savepoint.

    Notes
    -----
    Releasing the outermost savepoint commits standalone work. Inside an
    existing transaction, release does not commit the caller's changes.
    """
    name = f"pxsim_{next(_SAVEPOINTS)}"
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
        conn.execute(f"RELEASE {name}")
    except BaseException:
        # SQLite may already have rolled back an entire transaction on disk-full.
        if conn.in_transaction:
            conn.execute(f"ROLLBACK TO {name}")
            conn.execute(f"RELEASE {name}")
        raise


def effective_chunk_bytes(conn: sqlite3.Connection, requested: int) -> int:
    """Choose a chunk size with room for compression and SQLite record overhead.

    Parameters
    ----------
    conn : sqlite3.Connection
        Connection supplying the active SQLite length limit.
    requested : int
        Desired decoded bytes per chunk, from 1 through MAX_CHUNK_BYTES.

    Returns
    -------
    int
        The requested size or a smaller size fitting the connection limit.

    Raises
    ------
    ValueError
        The requested size is outside the supported range.
    PayloadWriteError
        The connection limit cannot accommodate a chunk safely.
    """
    if not isinstance(requested, int) or not 1 <= requested <= MAX_CHUNK_BYTES:
        raise ValueError(f"chunk_bytes must be between 1 and {MAX_CHUNK_BYTES}")
    limit = conn.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)
    # LZ4 stores incompressible blocks verbatim plus small frame/block headers.
    # Half the available row budget is deliberately conservative.
    size = min(requested, (limit - 512) // 2)
    if size < 64:
        raise PayloadWriteError(f"SQLite length limit {limit} is too small for chunk storage")
    return size


def payload_info(conn: sqlite3.Connection, payload_id: int, serializer: str | None = None) -> PayloadInfo:
    """Validate payload metadata before allocating or decoding its bytes.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open version 2 archive connection.
    payload_id : int
        Identifier in ``pxsim_payloads``.
    serializer : str, optional
        Required serializer, if specified. Currently only ``pickle`` is supported.

    Returns
    -------
    tuple
        Serializer, codec, pickle protocol, chunk size, raw byte count,
        stored byte count, chunk count, and raw SHA-256 digest, in that order.

    Raises
    ------
    IntegrityError
        The payload is missing or its header is invalid.
    UnsupportedFormatError
        The serializer, codec, or pickle protocol is unsupported.
    """
    row = conn.execute(
        "SELECT serializer,codec,pickle_protocol,chunk_bytes,raw_bytes,stored_bytes,chunk_count,sha256 "
        "FROM pxsim_payloads WHERE id=?",
        (payload_id,),
    ).fetchone()
    if row is None:
        raise IntegrityError(f"Missing payload {payload_id}")
    kind, codec, protocol, size, raw, stored, count, digest = row
    if kind != "pickle" or codec != CODEC:
        raise UnsupportedFormatError(f"Unsupported payload encoding {kind!r}/{codec!r}")
    if serializer is not None and kind != serializer:
        raise IntegrityError(f"Payload {payload_id} must have serializer {serializer}")
    if (
        not all(isinstance(n, int) for n in (size, raw, stored, count))
        or not 1 <= size <= MAX_CHUNK_BYTES
        or min(raw, stored, count) < 0
        or not isinstance(digest, bytes)
        or len(digest) != 32
    ):
        raise IntegrityError(f"Invalid header for payload {payload_id}")
    if protocol is not None and (not isinstance(protocol, int) or not 0 <= protocol <= 5):
        raise UnsupportedFormatError(f"Unsupported pickle protocol {protocol}")
    return row


class ChunkWriter:
    """Consume serializer output through a bounded compression buffer.

    Parameters
    ----------
    conn : sqlite3.Connection
        Version 2 archive connection inside a caller-owned transaction.
    serializer : str
        Payload interpretation; currently ``pickle``.
    protocol : int, optional
        Pickle protocol from 0 through 5; omit when the source protocol is unknown.
    chunk_bytes : int, default CHUNK_BYTES
        Maximum decoded bytes per chunk, reduced for the SQLite length limit.

    Notes
    -----
    Call ``finish`` before committing. On any failure, roll back the enclosing
    transaction so no incomplete payload remains. The writer does not own or
    close the connection.
    """

    def __init__(
        self, conn: sqlite3.Connection, serializer: str, *, protocol: int | None = None, chunk_bytes: int = CHUNK_BYTES
    ) -> None:
        """Create an unfinished payload inside the caller transaction."""
        self.conn = conn
        self.size = effective_chunk_bytes(conn, chunk_bytes)
        self.id = conn.execute(
            "INSERT INTO pxsim_payloads VALUES(NULL,?,?,?,?,0,0,0,?)",
            (serializer, CODEC, protocol, self.size, bytes(32)),
        ).lastrowid
        self.buffer = bytearray()
        self.digest = hashlib.sha256()
        self.raw = self.stored = self.count = 0
        self.finished = False

    def write(self, data: Buffer) -> int:
        """Consume a contiguous buffer without copying it in full.

        Parameters
        ----------
        data : buffer
            Contiguous object supporting the buffer protocol, including pickle
            buffers. Data is consumed synchronously; no caller buffer is retained.

        Returns
        -------
        int
            Number of bytes consumed.

        Raises
        ------
        ValueError
            The writer has already been finalized.
        """
        if self.finished:
            raise ValueError("Payload writer is finished")
        view = memoryview(data).cast("B")
        length = len(view)
        while view:
            take = min(len(view), self.size - len(self.buffer))
            self.buffer.extend(view[:take])
            view = view[take:]
            if len(self.buffer) == self.size:
                self._emit()
        return length

    def _emit(self) -> None:
        """Compress the pending chunk and accumulate its size and digest."""
        if not self.buffer:
            return
        encoded = lz4.frame.compress(self.buffer)
        self.conn.execute(
            "INSERT INTO pxsim_chunks VALUES(?,?,?,?,?)",
            (self.id, self.count, len(self.buffer), hashlib.sha256(self.buffer).digest(), encoded),
        )
        self.digest.update(self.buffer)
        self.raw += len(self.buffer)
        self.stored += len(encoded)
        self.count += 1
        self.buffer.clear()

    def finish(self) -> int:
        """Flush the final chunk and persist aggregate size and digest metadata.

        Returns
        -------
        int
            Payload identifier for the caller to reference in a named entry.

        Notes
        -----
        Repeated calls return the same identifier. This does not commit.
        """
        if not self.finished:
            self._emit()
            self.conn.execute(
                "UPDATE pxsim_payloads SET raw_bytes=?,stored_bytes=?,chunk_count=?,sha256=? WHERE id=?",
                (self.raw, self.stored, self.count, self.digest.digest(), self.id),
            )
            self.finished = True
        return self.id


class ChunkReader(io.RawIOBase):
    """Verify and decode a payload while retaining at most one decoded chunk.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open version 2 archive connection, optionally read-only.
    payload_id : int
        Identifier of the payload to read.
    serializer : str, optional
        Expected serializer; a mismatch raises IntegrityError.

    Notes
    -----
    Individual chunks are verified when decoded. Aggregate counts and digest
    are verified at EOF; call ``finish`` if the consumer stops early.
    Closing releases the cursor without draining or closing the connection.
    Inherited unbounded ``read()`` can still allocate the entire payload.
    """

    def __init__(self, conn: sqlite3.Connection, payload_id: int, serializer: str | None = None) -> None:
        """Validate the header and open a bounded, ordered chunk cursor."""
        super().__init__()
        self.id = payload_id
        self.info = payload_info(conn, payload_id, serializer)
        self.size = self.info[3]
        # Do not fetch an arbitrarily large corrupt BLOB into Python memory.
        self.cursor = conn.execute(
            "SELECT ordinal,raw_bytes,sha256,length(data),"
            "CASE WHEN typeof(data)='blob' AND length(data)<=? THEN data ELSE NULL END "
            "FROM pxsim_chunks WHERE payload_id=? ORDER BY ordinal",
            (self.size * 2 + 128, payload_id),
        )
        self.buffer = memoryview(b"")
        self.digest = hashlib.sha256()
        self.raw = self.stored = self.count = 0
        self.eof = False

    def readable(self) -> bool:
        """Return True because this stream supports reading."""
        return True

    def _next(self) -> None:
        """Decode the next bounded frame or verify the aggregate at EOF."""
        row = self.cursor.fetchone()
        if row is None:
            if (self.raw, self.stored, self.count, self.digest.digest()) != self.info[4:]:
                raise IntegrityError(f"Payload {self.id} size, chunk count, or checksum mismatch")
            self.eof = True
            return
        ordinal, length, digest, stored, encoded = row
        if (
            ordinal != self.count
            or not isinstance(length, int)
            or not 0 < length <= self.size
            or encoded is None
            or stored > self.size * 2 + 128
        ):
            raise IntegrityError(f"Invalid payload {self.id} chunk {self.count}")
        decoder = lz4.frame.LZ4FrameDecompressor()
        try:
            raw = decoder.decompress(encoded, max_length=length + 1)
        except (RuntimeError, ValueError) as error:
            raise IntegrityError(f"Invalid LZ4 payload {self.id} chunk {ordinal}") from error
        if not decoder.eof or decoder.unused_data or len(raw) != length or hashlib.sha256(raw).digest() != digest:
            raise IntegrityError(f"Payload {self.id} chunk {ordinal} checksum or frame mismatch")
        self.digest.update(raw)
        self.raw += length
        self.stored += stored
        self.count += 1
        self.buffer = memoryview(raw)

    def readinto(self, target: Buffer) -> int:
        """Fill a writable buffer while retaining only one decoded chunk.

        Parameters
        ----------
        target : buffer
            Writable contiguous buffer to fill.

        Returns
        -------
        int
            Bytes copied, or zero at EOF or for an empty target.

        Raises
        ------
        IntegrityError
            A decoded chunk or the aggregate payload fails validation.
        ValueError
            The reader is closed.
        """
        self._checkClosed()
        target = memoryview(target).cast("B")
        done = 0
        while done < len(target):
            if not self.buffer:
                if self.eof:
                    break
                self._next()
                if self.eof:
                    break
            take = min(len(target) - done, len(self.buffer))
            target[done : done + take] = self.buffer[:take]
            self.buffer = self.buffer[take:]
            done += take
        return done

    def finish(self) -> None:
        """Drain unread bytes and verify aggregate counts and digest.

        Raises
        ------
        IntegrityError
            Chunk structure, byte totals, or checksums do not match.

        Notes
        -----
        Use this after a serializer stops reading to validate the entire payload.
        Repeated calls at EOF have no effect.
        """
        scratch = bytearray(self.size)
        while self.readinto(scratch):
            pass

    def close(self) -> None:
        """Release the cursor and buffered bytes without validating unread data."""
        if not self.closed:
            self.cursor.close()
            self.buffer = memoryview(b"")
        super().close()
