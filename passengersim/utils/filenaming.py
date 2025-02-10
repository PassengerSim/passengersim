import datetime
import os
import pathlib
import re
import time


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
    if timestamp is None or timestamp is True:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
    elif isinstance(timestamp, float):
        timestamp = datetime.datetime.fromtimestamp(timestamp).strftime("%Y%m%d-%H%M%S")
    elif isinstance(timestamp, time.struct_time):
        timestamp = time.strftime("%Y%m%d-%H%M%S", timestamp)
    elif isinstance(timestamp, datetime.datetime):
        timestamp = timestamp.strftime("%Y%m%d-%H%M%S")
    else:
        raise ValueError(f"Invalid timestamp type: {type(timestamp)}")
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

    while filename.exists():
        t0, t1 = timestamp.split("-")
        t1 = str(int(t1) + 1)
        timestamp = f"{t0}-{t1}"
        filename = filename.with_suffix(f".{timestamp}" + suffix)

    if make_dirs:
        filename.parent.mkdir(parents=True, exist_ok=True)

    return filename
