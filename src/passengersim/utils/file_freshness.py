import os
import pathlib

import yaml

PathLike = pathlib.Path | str | os.PathLike[str]


def check_modification_times(filenames: list[PathLike] | PathLike, cache_file: PathLike) -> str | None:
    """
    Check the modification times of a list of files against a cache file.

    Parameters
    ----------
    filenames : list[PathLike] or PathLike
        List of files to check.
    cache_file : PathLike
        Cache file to check against.

    Returns
    -------
    str or None
        Return "outdated" if any of the files have been modified since the cache file
        was last modified, or "missing" if the cache file does not exist, and None
        otherwise.
    """
    filenames = preprocess_filenames(filenames)
    cache_file = pathlib.Path(cache_file)
    if not cache_file.exists():
        return "missing"
    cache_time = cache_file.stat().st_mtime
    for filename in filenames:
        if isinstance(filename, str) and "\n" in filename:
            # raw YAML content, assume it is always not cached
            return "raw"
        if filename.stat().st_mtime > cache_time:
            return "outdated"
    return None


def preprocess_filenames(
    filenames: list[PathLike] | PathLike, expand_includes: bool = True
) -> list[pathlib.Path | str]:
    """
    Preprocess a list of filenames, expanding user directories and include directives.

    Parameters
    ----------
    filenames : list[str] or str
        List of filenames to preprocess.

    Returns
    -------
    list[pathlib.Path or str]
        List of preprocessed filenames as pathlib.Path objects, or raw YAML
        content as a multi-line string.
    """

    if isinstance(filenames, pathlib.Path | str | os.PathLike):
        filenames = [filenames]
    result = []
    for filename in filenames:
        if isinstance(filename, str) and "\n" in filename:
            # treat as raw YAML content
            result.append(filename)
            continue
        path = pathlib.Path(filename).expanduser()

        if path.is_dir() and (path / "__main__.yaml").exists():
            path = path / "__main__.yaml"

        if path.is_file() and path.name == "__main__.yaml":
            parent_path = path.parent
        else:
            parent_path = None

        # if the total file size is less than 100KB, load the file and check if it
        # has an "include" directive, and if so, preprocess the included file(s) as well
        if expand_includes and path.exists() and path.stat().st_size < 100 * 1024:
            with open(path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "include" in data and len(data) == 1:
                # the only key is "include", so expand it
                included_files = data["include"]
                if isinstance(included_files, str):
                    included_files = [included_files]
                included_files = [pathlib.Path(i) for i in included_files]
                if parent_path is not None:
                    included_files = [(i if i.is_absolute() else parent_path.joinpath(i)) for i in included_files]
                result.extend(preprocess_filenames(included_files))

        result.append(path)
    return result
