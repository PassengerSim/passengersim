import importlib.resources
import re
from pathlib import Path

from passengersim import Config


def _network_root(package_name: str) -> Path:
    """Locate the network data directory.

    Uses :func:`importlib.resources.files` so the lookup works correctly in
    both installed-wheel and editable/development layouts:

    - Installed wheel:  ``<site-packages>/<package_name>/configs/``
    - Editable / source tree:  ``<repo-root>/configs/``
    """
    pkg_files = importlib.resources.files(package_name)

    # Primary: configs/ bundled inside the installed package.
    installed = pkg_files / "configs"
    if installed.is_dir():
        return Path(installed)  # type: ignore[arg-type]

    # Development fallback: configs/ lives at the repo root, which is two
    # directory levels above src/<package_name>/ (the path that pkg_files resolves to
    # in an editable install).
    source = Path(pkg_files).parent.parent / "configs"  # type: ignore[arg-type]
    if source.is_dir():
        return source

    raise FileNotFoundError(
        f"Cannot locate the {package_name} network data directory. Looked in {installed} and {source}."
    )


def _maybe_int(i):
    try:
        return int(i)
    except Exception:
        return i


def _find_latest(base_dir: Path, required_patch: str | None = None) -> Path:
    """Find the latest version subdirectory of base_dir.

    This function will scan all immediate subdirectories of base_dir and
    collect all such directories matching the pattern "v{major}.{minor}.{patch}",
    where major, minor, and patch are integers. It will then determine the latest
    version among these directories and return the path to that directory.
    """

    if required_patch:
        version_pattern = re.compile(rf"^v(\d+)\.(\d+)\.({required_patch})$", re.IGNORECASE)
    else:
        version_pattern = re.compile(r"^v(\d+)\.(\d+)\.(\w+)$", re.IGNORECASE)
    candidates: list[tuple[tuple[int, int, int | str], Path]] = []

    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        m = version_pattern.match(entry.name)
        if m:
            version_tuple = (int(m.group(1)), int(m.group(2)), _maybe_int(m.group(3)))
            candidates.append((version_tuple, entry))

    if not candidates:
        raise FileNotFoundError(f"No versioned subdirectories (v{{major}}.{{minor}}.{{patch}}) found in {base_dir}.")

    # Return the path corresponding to the highest version tuple.
    _, latest_path = max(candidates, key=lambda item: item[0])
    return latest_path


def find_config(name: str = "latest", *, variant: str | None = None, package_name: str) -> Path:
    """Fine the matching network config file bundled in package."""
    root = _network_root(package_name)
    name_lower = str(name).lower()
    if name_lower == "latest":
        name = _find_latest(root, required_patch=variant)
    elif name_lower.endswith("/latest"):
        name = _find_latest(root / str(name)[:-7], required_patch=variant)
    elif "/latest." in name_lower:
        leading_path, required_patch = str(name).split("/latest.")
        name = _find_latest(root / leading_path, required_patch=required_patch)
    elif name_lower.startswith("latest."):
        required_patch = name_lower[7:]
        name = _find_latest(root, required_patch=required_patch)
    return root / name


def load_config(name: str = "latest", *, variant: str | None = None, package_name: str) -> Config:
    """Load the network config file bundled with this package."""
    found = find_config(name, variant=variant, package_name=package_name)
    print(f"Loading network config file {found!s}")
    cfg = Config.from_yaml(found)
    cfg.tags["package"] = package_name
    cfg.tags["version"] = found.name
    return cfg
