import pathlib
import re
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
            assert re.match(
                r"subdir/hello-world\.200902\d{2}-\d{4}30\.html", str(f1[".html"])
            )
            assert re.match(
                r"subdir/hello-world\.200902\d{2}-\d{4}30\.log", str(f1[".log"])
            )
            assert re.match(r"200902\d{2}-\d{4}30", f1["timestamp"])
            f1[".html"].write_text("*")
            f2 = filenames_with_timestamp(
                "subdir/hello-world", ["html", "log"], timestamp=1234567890
            )
            f2[".log"].write_text("*")
            assert re.match(
                r"subdir/hello-world\.200902\d{2}-\d{4}31\.html", str(f2[".html"])
            )
            assert re.match(
                r"subdir/hello-world\.200902\d{2}-\d{4}31\.log", str(f2[".log"])
            )
            assert re.match(r"200902\d{2}-\d{4}31", f2["timestamp"])
