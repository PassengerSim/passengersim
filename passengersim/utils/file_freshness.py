import pathlib


def check_modification_times(
    filenames: list[pathlib.Path] | pathlib.Path, cache_file: pathlib.Path
) -> bool:
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
    bool
        Return True if any of the files have been modified since the cache file
        was last modified, and False otherwise.
    """
    if isinstance(filenames, pathlib.Path | str):
        filenames = [filenames]
    filenames = [pathlib.Path(filename) for filename in filenames]
    cache_file = pathlib.Path(cache_file)
    if not cache_file.exists():
        return True
    cache_time = cache_file.stat().st_mtime
    for filename in filenames:
        if filename.stat().st_mtime > cache_time:
            return True
    return False
