import datetime
import os
import pathlib
import re
import time
from collections.abc import Sequence


def _format_timestamp(
    timestamp: float | time.struct_time | datetime.datetime | None,
) -> str:
    if timestamp is None or timestamp is True:
        return time.strftime("%Y%m%d-%H%M%S")
    elif isinstance(timestamp, float | int):
        return datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d-%H%M%S")
    elif isinstance(timestamp, time.struct_time):
        return time.strftime("%Y%m%d-%H%M%S", timestamp)
    elif isinstance(timestamp, datetime.datetime):
        return timestamp.strftime("%Y%m%d-%H%M%S")
    elif timestamp is False:
        return "no-timestamp"
    else:
        raise ValueError(f"Invalid timestamp type: {type(timestamp)}")


def filename_with_timestamp(
    filename: str | os.PathLike,
    *,
    timestamp: float | time.struct_time | datetime.datetime | None = None,
    suffix: str | None = None,
    make_dirs: bool = False,
) -> pathlib.Path:
    """
    Add a timestamp to a filename.

    The resulting filename will be the original filename with a timestamp added.
    If the stamped filename already exists, the timestamp will be incremented
    until a unique filename is found. If the provided filename already has a
    timestamp, it will be replaced.

    Parameters
    ----------
    filename : str or os.PathLike
        The filename to which to add a timestamp.
    timestamp : float, time.struct_time, datetime.datetime, or None
        The timestamp to add.  If None, the current time will be used.
    suffix : str or None, optional
        The suffix to use.  If None, the suffix of the original filename will be
        used.
    make_dirs : bool, optional
        If True, create any necessary intermediate directories, so that a file
        can be created at the returned path.

    Returns
    -------
    pathlib.Path
        The new filename with the timestamp added.
    """
    timestamp = _format_timestamp(timestamp)
    filename = pathlib.Path(filename)
    if suffix is None:
        suffix = filename.suffix

    filename_str = str(filename)
    # check if the filename already has a timestamp
    pattern = r"\.\d{8}-\d{6}" + re.escape(suffix) + r"$"
    match = re.search(pattern, filename_str)
    if match:
        filename_str = re.sub(pattern, f".{timestamp}{suffix}", filename_str)
        filename = pathlib.Path(filename_str)
    else:
        filename = filename.with_suffix(f".{timestamp}" + suffix)

    while filename.exists() and timestamp != "no-timestamp":
        t0, t1 = timestamp.split("-")
        t1 = str(int(t1) + 1)
        timestamp = f"{t0}-{t1}"
        filename = filename.with_suffix(f".{timestamp}" + suffix)

    if make_dirs:
        filename.parent.mkdir(parents=True, exist_ok=True)

    return filename


def filenames_with_timestamp(
    filename: str | os.PathLike,
    suffix: Sequence[str],
    *,
    timestamp: float | time.struct_time | datetime.datetime | None = None,
    make_dirs: bool = False,
) -> dict[str, pathlib.Path | str]:
    """
    Add a consistent timestamp to multiple filenames sharing a common stem.

    The resulting filenames will be the original filename stem with
    a timestamp and then each suffix added. If any of the stamped filenames
    already exists, the timestamp will be incremented until a unique set of
    filenames is found. If the provided filename stem already has a
    timestamp, it will be replaced.

    Parameters
    ----------
    filename : str or os.PathLike
        The filename stem to which to add a timestamp and the suffixes.
    suffix : list[str]
        The suffixes to use.  Unlike the simple `filenames_with_timestamp`
        function, these must be provided.
    timestamp : float, time.struct_time, datetime.datetime, or None
        The timestamp to add.  If None, the current time will be used.
    make_dirs : bool, optional
        If True, create any necessary intermediate directories, so that a file
        can be created at the returned path.

    Returns
    -------
    dict[str, pathlib.Path]
        The new filenames with the timestamp added.  The keys are the suffixes,
        and the values are the new filenames with that suffix. The keys will
        also include "timestamp" with the timestamp value.
    """
    # ensure all suffixes are strings that start with a period
    suffix = [s if s.startswith(".") else f".{s}" for s in suffix]

    timestamp = _format_timestamp(timestamp)
    filename = pathlib.Path(filename)
    filename_str = str(filename)
    # check if the filename already has a timestamp ...
    for s in suffix:
        # ... with any of the suffixes
        pattern = r"\.\d{8}-\d{6}" + re.escape(s) + r"$"
        match = re.search(pattern, filename_str)
        if match:
            filename_str = re.sub(pattern, "", filename_str)
            filename = pathlib.Path(filename_str)
    ## ... or without any other suffix
    pattern = r"\.\d{8}-\d{6}" + r"$"
    match = re.search(pattern, filename_str)
    if match:
        filename_str = re.sub(pattern, "", filename_str)
        filename = pathlib.Path(filename_str)

    filenames = {s: filename.with_suffix(f".{timestamp}" + s) for s in suffix}

    while any([fi.exists() for fi in filenames.values()]) and timestamp != "no-timestamp":
        t0, t1 = timestamp.split("-")
        t1 = str(int(t1) + 1)
        timestamp = f"{t0}-{t1}"
        filenames = {s: filename.with_suffix(f".{timestamp}" + s) for s in suffix}

    if make_dirs:
        filename.parent.mkdir(parents=True, exist_ok=True)

    filenames["timestamp"] = timestamp
    return filenames


def make_parent_directory(filename: os.PathLike, git_ignore_things: list[str] | bool | None = None) -> None:
    """Create any necessary intermediate directories."""
    filename = pathlib.Path(filename)
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)
        if git_ignore_things:
            with open(filename.parent / ".gitignore", "a") as f:
                f.write(".gitignore\n")  # ignore this file itself
                if git_ignore_things is True:
                    f.write("**\n")
                else:
                    for thing in git_ignore_things:
                        f.write(f"{thing}\n")
