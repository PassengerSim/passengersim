import pickle
import sqlite3
from collections.abc import MutableMapping

try:
    import lz4.frame
except ImportError:
    lz4 = None


class KVStore(MutableMapping):
    def __init__(self, db_path=":memory:"):
        if lz4 is None:
            raise ImportError(
                "lz4 is not installed, but required for this file storage format."
            )
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS kvstore (
                    key TEXT PRIMARY KEY,
                    value BLOB
                )
            """)

    def __setitem__(self, key, value):
        serialized_value = pickle.dumps(value)
        compressed_value = lz4.frame.compress(serialized_value)
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO kvstore (key, value)
                VALUES (?, ?)
            """,
                (key, compressed_value),
            )

    def __getitem__(self, key):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT value FROM kvstore WHERE key = ?
        """,
            (key,),
        )
        row = cursor.fetchone()
        if row:
            compressed_value = row[0]
            serialized_value = lz4.frame.decompress(compressed_value)
            return pickle.loads(serialized_value)
        raise KeyError(key)

    def __delitem__(self, key):
        with self.conn:
            self.conn.execute(
                """
                DELETE FROM kvstore WHERE key = ?
            """,
                (key,),
            )

    def close(self):
        self.conn.close()

    def get_sizes(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, length(value) as size FROM kvstore")
        return dict(cursor.fetchall())

    def __len__(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM kvstore")
        return cursor.fetchone()[0]

    def __iter__(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT key FROM kvstore")
        for row in cursor.fetchall():
            yield row[0]
