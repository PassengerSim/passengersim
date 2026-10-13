"""Process and thread guards for parent-owned result connections."""

import sqlite3

from .database import Database


class RetainedConnection(sqlite3.Connection):
    """Keep storage alive and reject inherited raw connection use after fork."""

    def __getattribute__(self, name):
        if not name.startswith("_"):
            owner = super().__getattribute__("__dict__").get("_result_store")
            if owner is not None:
                owner.check_access()
        return super().__getattribute__(name)

    def close(self):
        """Explicit raw close has the same alias/cleanup semantics as result.close."""
        owner = self.__dict__.get("_result_store")
        if (
            owner is not None
            and owner._database is not None
            and owner._database._connection is self
            and not owner._closing
        ):
            owner.close()
        else:
            super().close()

    def cursor(self, factory=None):
        return super().cursor(factory or RetainedCursor)

    def __enter__(self):
        self._result_store.check_access()
        return super().__enter__()

    def __exit__(self, *args):
        self._result_store.check_access()
        return super().__exit__(*args)

    def __call__(self, *args, **kwargs):
        self._result_store.check_access()
        return super().__call__(*args, **kwargs)

    def execute(self, *args, **kwargs):
        return self.cursor().execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self.cursor().executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        return self.cursor().executescript(*args, **kwargs)


class RetainedCursor(sqlite3.Cursor):
    """Apply the same ownership guard to a cursor kept across a fork."""

    def __getattribute__(self, name):
        if not name.startswith("_"):
            super().__getattribute__("connection")._result_store.check_access()
        return super().__getattribute__(name)

    def __iter__(self):
        self.connection._result_store.check_access()
        return super().__iter__()

    def __next__(self):
        self.connection._result_store.check_access()
        return super().__next__()


class RetainedDatabase(Database):
    """Database alias whose explicit close also closes its shared storage owner."""

    def __getattribute__(self, name):
        if not name.startswith("_"):
            owner = super().__getattribute__("__dict__").get("_result_store")
            if owner is not None:
                owner.check_access()
        return super().__getattribute__(name)

    def close(self):
        self._result_store.close()

    def __del__(self):
        # The shared owner handles final-reference cleanup without committing.
        # Routing Database.__del__ through raw.close would accidentally commit.
        pass

    def open(self, filename=None):
        raise ConnectionError("A retained result cannot be reopened; load or run a new result")
