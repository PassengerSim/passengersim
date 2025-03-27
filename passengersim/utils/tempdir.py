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
