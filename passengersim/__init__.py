try:
    # before loading any other subpackages, first try to
    # load declarations of all the usual RmStep classes
    from passengersim_core import carrier as rm  # noqa: F401
except ImportError:
    raise

from ._version import __version__, __version_tuple__
from .cli.info import info  # noqa: F401
from .config import Config, OptionalPath
from .driver import Simulation
from .mp_driver2 import MultiSimulation
from .summaries import SimulationTables
from .summary import SummaryTables
from .utils.colors import passengersim_colors as _colors  # noqa: F401
from .utils.import_tool import import_from_path

__all__ = [
    "Config",
    "Simulation",
    "MultiSimulation",
    "OptionalPath",
    "SummaryTables",
    "SimulationTables",
    "demo_network",
    "import_from_path",
    "__version__",
    "__version_tuple__",
]


def demo_network(name: str):
    import importlib.resources

    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    # note: on macOS by default the file system is not case sensitive, but
    # on Linux it is. All the network files are lowercase, so we can get around
    # this by forcing the name to lowercase.
    return (
        importlib.resources.files(__package__)
        .joinpath("networks")
        .joinpath(name.lower())
    )


def versions(verbose=False):
    """Print the versions"""
    print(f"passengersim {__version__}")
    import passengersim_core

    if verbose:
        print(
            f"passengersim.core {passengersim_core.__version__} "
            f"(expires {passengersim_core.build_expiration()})"
        )
    else:
        print(f"passengersim.core {passengersim_core.__version__}")


def logging(level=None):
    from ._logging import log_to_console

    log_to_console(level)


def _has_file_extensions(path: str, ext: list[str]) -> bool:
    path = str(path)
    lowercase_path = path.lower()
    return any(lowercase_path.endswith(e) for e in ext)

def from_file(path: str):
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
        return SimulationTables.from_file(path)
    elif _has_file_extensions(path, [".yaml", ".yml", ".yml.gz", ".yaml.gz", ".yml.lz4", ".yaml.lz4"]):
        return Config.from_yaml(path)
    else:
        raise ValueError("Unknown file type")