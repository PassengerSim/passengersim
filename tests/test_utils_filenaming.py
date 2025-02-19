import pathlib
import tempfile
from contextlib import chdir

from passengersim.utils.filenaming import filenames_with_timestamp


def test_filenaming_consistent_timestamps():
    with tempfile.TemporaryDirectory() as td:
        tp = pathlib.Path(td)
        with chdir(tp):
            f1 = filenames_with_timestamp(
                "subdir/hello-world",
                ["html", "log"],
                timestamp=1234567890,
                make_dirs=True,
            )
            assert f1 == {
                ".html": pathlib.Path("subdir/hello-world.20090213-173130.html"),
                ".log": pathlib.Path("subdir/hello-world.20090213-173130.log"),
                "timestamp": "20090213-173130",
            }
            f1[".html"].write_text("*")
            f2 = filenames_with_timestamp(
                "subdir/hello-world", ["html", "log"], timestamp=1234567890
            )
            f2[".log"].write_text("*")
            assert f2 == {
                ".html": pathlib.Path("subdir/hello-world.20090213-173131.html"),
                ".log": pathlib.Path("subdir/hello-world.20090213-173131.log"),
                "timestamp": "20090213-173131",
            }
