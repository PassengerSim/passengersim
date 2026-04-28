import os
import pathlib
import tempfile


class TemporaryDirectory(tempfile.TemporaryDirectory):
    def __fspath__(self):
        return self.name

    @property
    def path(self) -> pathlib.Path:
        return pathlib.Path(self.name)

    def joinpath(self, *other) -> pathlib.Path:
        return pathlib.Path(self.name).joinpath(*other)


class MaybeTemporaryDirectory(TemporaryDirectory):
    """A temporary directory as needed.

    If an existing directory is provided, it will not be deleted on exit.
    """

    def __init__(self, existing: str | pathlib.Path | os.PathLike | None = None):
        self._existing = existing
        if existing is None:
            super().__init__(ignore_cleanup_errors=True)
        else:
            self.name = str(self._existing)

    def __enter__(self):
        if self._existing is None:
            return super().__enter__()
        else:
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._existing is None:
            return super().__exit__(exc_type, exc_value, traceback)
        else:
            # Do not delete the existing directory
            return False

    def __del__(self):
        if self._existing is None:
            try:
                self.cleanup()
            except Exception:
                # Ignore errors on deletion
                pass
        # else: do nothing, we don't delete the existing directory
