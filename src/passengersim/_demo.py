import importlib.resources
import pathlib


def demo_network(name: str) -> pathlib.Path:
    """
    Load a demo network configuration file.

    A few demo networks are bundled with the PassengerSim package, and are always
    available to users.

    Parameters
    ----------
    name : str
        The name of the network file, with or without the .yaml extension.
        If the extension is not provided, it will be added automatically.

    Returns
    -------
    pathlib.Path
        Path object pointing to the network configuration file.

    Notes
    -----
    The network name is converted to lowercase to handle case sensitivity
    differences between macOS (case-insensitive) and Linux (case-sensitive)
    file systems. All packaged network files are stored in lowercase.
    """
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    # note: on macOS by default the file system is not case sensitive, but
    # on Linux it is. All the network files are lowercase, so we can get around
    # this by forcing the name to lowercase.
    return importlib.resources.files(__package__).joinpath("networks").joinpath(name.lower())


def demo_output(name: str) -> pathlib.Path:
    """
    Load a demo output file.

    A few demo output files are bundled with the PassengerSim package, and are always
    available to user, without needing to load or run a simulation.

    Parameters
    ----------
    name : str
        The name of the output file, with or without the .pxsim extension.
        If the extension is not provided, it will be added automatically.

    Returns
    -------
    pathlib.Path
        Path object pointing to the output file.

    Notes
    -----
    The output name is converted to lowercase to handle case sensitivity
    differences between macOS (case-insensitive) and Linux (case-sensitive)
    file systems. All output files are stored in lowercase.
    """
    if not name.endswith(".pxsim"):
        name = f"{name}.pxsim"
    return importlib.resources.files(__package__).joinpath("networks").joinpath(name.lower())
