import importlib.util
import pathlib
import sys


def import_from_path(file_path, module_name=None, sys_modules: bool = False):
    """
    Import a module from a file path.

    Use this tool to import a module from a file path.  This is useful when you want to
    import a module that is not in the current working directory, and not necessarily
    in the Python path.

    Parameters
    ----------
    file_path : str
        The path to the file to import.
    module_name : str, optional
        The name of the module to import.  If not provided, the module name will be the
        stem of the file path.
    sys_modules : bool, default False
        If True, the module will be added to sys.modules. This is useful if you want to
        import the module multiple times and have it be the same object. This requires
        that the module name is unique, just like a normal import.

    Returns
    -------
    module : module
        The imported module.
    """
    if module_name is None:
        module_name = pathlib.Path(file_path).stem
    if sys_modules and module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if sys_modules:
        sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
