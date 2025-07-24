import pathlib


def check_modification_times(filenames: list[pathlib.Path] | pathlib.Path, cache_file: pathlib.Path) -> str | None:
    """
    Check the modification times of a list of files against a cache file.

    Parameters
    ----------
    filenames : list[pathlib.Path] or pathlib.Path
        List of files to check.
    cache_file : pathlib.Path
        Cache file to check against.

    Returns
    -------
    str or None
        Return "outdated" if any of the files have been modified since the cache file
        was last modified, or "missing" if the cache file does not exist, and None
        otherwise.
    """
    if isinstance(filenames, pathlib.Path | str):
        filenames = [filenames]
    filenames = [pathlib.Path(filename) for filename in filenames]
    cache_file = pathlib.Path(cache_file)
    if not cache_file.exists():
        return "missing"
    cache_time = cache_file.stat().st_mtime
    for filename in filenames:
        if filename.stat().st_mtime > cache_time:
            return "outdated"
    return None
