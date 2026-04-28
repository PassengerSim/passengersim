from . import rm, transforms
from ._demo import demo_network, demo_output
from ._version import __version__, __version_tuple__
from .cli.info import info  # noqa: F401
from .config import Config, OptionalPath
from .driver import Simulation
from .file_ops import from_file
from .mp_driver import MultiSimulation
from .summaries import SimulationTables
from .utils.colors import passengersim_colors as _colors  # noqa: F401
from .utils.import_tool import import_from_path

__all__ = [
    "Config",
    "Simulation",
    "MultiSimulation",
    "OptionalPath",
    "SimulationTables",
    "demo_network",
    "demo_output",
    "from_file",
    "import_from_path",
    "__version__",
    "__version_tuple__",
    "transforms",
    "rm",
]


def versions(verbose=False):
    """Print the versions"""
    print(f"passengersim {__version__}")
    import passengersim_core

    if verbose:
        print(f"passengersim.core {passengersim_core.__version__} (expires {passengersim_core.build_expiration()})")
    else:
        print(f"passengersim.core {passengersim_core.__version__}")


def logging(level=None):
    from ._logging import log_to_console

    log_to_console(level)
