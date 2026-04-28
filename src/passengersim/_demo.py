import importlib.resources
import pathlib


def demo_network(name: str) -> pathlib.Path:
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    # note: on macOS by default the file system is not case sensitive, but
    # on Linux it is. All the network files are lowercase, so we can get around
    # this by forcing the name to lowercase.
    return importlib.resources.files(__package__).joinpath("networks").joinpath(name.lower())


def demo_output(name: str) -> pathlib.Path:
    if not name.endswith(".pxsim"):
        name = f"{name}.pxsim"
    return importlib.resources.files(__package__).joinpath("networks").joinpath(name.lower())
