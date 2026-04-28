import pathlib

import yaml

from passengersim._types import PathLike


def expand_small_file_with_include_key(filename: PathLike | list[PathLike]) -> list[pathlib.Path]:
    """
    For small files, load them as YAML and check for an "include" key.

    If found, return a list of the included files.
    """
    # if filename is a list or tuple, process each element
    if isinstance(filename, (list | tuple)):
        result = []
        for f in filename:
            result.extend(expand_small_file_with_include_key(f))
        return result

    # ensure filename is a pathlib.Path
    filename = pathlib.Path(filename).expanduser()

    # If the provided filename is a directory that contains a file named "__main__.yaml",
    # use that file instead.
    if filename.is_dir() and (filename / "__main__.yaml").exists():
        filename = filename / "__main__.yaml"

    # check file size
    if filename.stat().st_size > 10240:  # 10KB
        # for larger files, assume no includes
        return [filename]

    with open(filename) as f:
        content = yaml.safe_load(f)

    if "include" in content:
        includes = content["include"]
        if isinstance(includes, str):
            includes = [includes]
        if isinstance(includes, list | tuple):
            subs = []
            for i in includes:
                if isinstance(i, str):
                    subs.append(filename.parent.joinpath(i))
                else:
                    raise ValueError(f"Invalid include entry in {filename}: {i}")
            return [filename] + subs

    return [filename]
