import importlib.util
import pathlib
import sys
import tomllib

temp_dir = None


def import_from_path(file_path, module_name=None, sys_modules: bool = True):
    """
    Import a module from a file path.

    Use this tool to import a module from a file path.  This is useful when you want to
    import a module that is not in the current working directory, and not necessarily
    in the Python path.  Both single-file modules and packages (directories containing
    an ``__init__.py`` file) are supported.

    Parameters
    ----------
    file_path : str or path-like
        The path to the file or package directory to import.
    module_name : str, optional
        The name of the module to import.  If not provided, the module name will be the
        stem of the file path (for single-file modules) or the directory name (for
        packages).
    sys_modules : bool, default True
        If True, the module will be added to sys.modules. This is useful if you want to
        import the module multiple times and have it be the same object. This requires
        that the module name is unique, just like a normal import.

    Returns
    -------
    module : module
        The imported module.
    """
    path = pathlib.Path(file_path)

    # Resolve the actual file to load and determine whether this is a package.
    if path.is_dir():
        # Treat a directory as a package if it contains an __init__.py file.
        init_file = path / "__init__.py"
        if not init_file.exists():
            # it IS a directory, but it does NOT contain an __init__.py file.
            # WAIT! One last check: is there a pyproject.toml file?
            pyproject_file = path / "pyproject.toml"
            if pyproject_file.exists():
                # The pyproject.toml file exists, so we will try to find packages in it.
                with pyproject_file.open("rb") as f:
                    pyproject = tomllib.load(f)
                # look for packages using hatch.
                packages = (
                    pyproject.get("tool", {})
                    .get("hatch", {})
                    .get("build", {})
                    .get("targets", {})
                    .get("wheel", {})
                    .get("packages", [])
                )
                if len(packages) > 1:
                    raise ImportError(
                        f"Directory '{path}' does not contain an '__init__.py' file and "
                        f"contains multiple packages in 'pyproject.toml', so it cannot be imported as one package."
                    )
                if len(packages) == 1:
                    return import_from_path(path / packages[0], module_name=module_name, sys_modules=sys_modules)
            raise ImportError(
                f"Directory '{path}' does not contain an '__init__.py' file and cannot be imported as a package."
            )
        if module_name is None:
            module_name = path.name
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_file,
            submodule_search_locations=[str(path)],
        )
    else:
        # Single-file module.
        if module_name is None:
            module_name = path.stem
        spec = importlib.util.spec_from_file_location(module_name, path)

    if sys_modules and module_name in sys.modules:
        return sys.modules[module_name]

    module = importlib.util.module_from_spec(spec)

    # The module must be in sys.modules before exec_module is called so that
    # relative imports inside the package (e.g. "from .__about__ import ...")
    # can resolve correctly.  If the caller did not request sys_modules
    # persistence, we remove the entry again after loading.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Clean up the partially-loaded module on failure.
        sys.modules.pop(module_name, None)
        raise

    if not sys_modules:
        sys.modules.pop(module_name, None)

    return module
