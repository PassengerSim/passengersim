import os
import pathlib

from passengersim.config import Config
from passengersim.driver import get_default_summarizer


def _has_file_extensions(path: str, ext: list[str]) -> bool:
    path = str(path)
    lowercase_path = path.lower()
    return any(lowercase_path.endswith(e) for e in ext)


def from_file(path: str | pathlib.Path | os.PathLike[str]) -> Config:
    """Load a PassengerSim file.

    The type of file is determined by the extension.  The following file types
    are supported:

    - `.pxsim`: SimulationTables
    - `.yaml`, `.yml`, or either also with `.gz` or `.lz4`: Config

    Parameters
    ----------
    path : str
        Path to the file to load. The file type is determined by the extension.
    """
    if _has_file_extensions(path, [".pxsim"]):
        return get_default_summarizer().from_file(path)
    elif _has_file_extensions(path, [".yaml", ".yml", ".yml.gz", ".yaml.gz", ".yml.lz4", ".yaml.lz4"]):
        return Config.from_yaml(path)
    else:
        pth = pathlib.Path(path)
        if pth.is_dir() and (pth / "__main__.yaml").exists():
            return Config.from_yaml(pth / "__main__.yaml")
        raise ValueError("Unknown file type")
