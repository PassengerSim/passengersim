from __future__ import annotations

import hashlib
import pathlib
from typing import TYPE_CHECKING, TypeVar

from passengersim import Config, MultiSimulation, Simulation
from passengersim.summaries.generic import GenericSimulationTables

if TYPE_CHECKING:
    from pydantic.main import IncEx

SimulationTablesT = TypeVar("SimulationTablesT", bound=GenericSimulationTables)


def generate_hash(
    cfg, *, include: IncEx | None = None, exclude: IncEx | None = None, extra: list[str] | None = None
) -> str:
    """Generate a hash of the Config object."""
    if exclude is None:
        exclude = {
            "tags": True,
            "raw_license_certificate": True,
            "outputs": True,
        }
    j = cfg.model_dump_json(include=include, exclude=exclude)
    h = hashlib.sha256(j.encode())
    for el in extra:
        h.update(el.encode())
    return h.hexdigest()


def cache_run[SimulationTablesT: GenericSimulationTables](
    sim: Simulation | MultiSimulation,
    cache_dir: str | pathlib.Path,
    *,
    summarizer: type[SimulationTablesT] | SimulationTablesT | None = None,
    key: str = "",
    key_check: str = "",
    **kwargs,
) -> SimulationTablesT:
    """Cache the run configuration and return the cache key."""

    cfg: Config = sim.config

    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    extra = []
    if summarizer is not None:
        extra.append(summarizer.__name__)
        extra.append(summarizer.__module__)

    cache_key = key or key_check or generate_hash(cfg, extra=extra)

    if key_check:
        computed_key = generate_hash(cfg, extra=extra)
        if computed_key != key_check:
            print(f" computed hash {computed_key}")
            print(f" provided hash {key_check}")
            raise ValueError("computed hash does not match provided hash")

    cache_path = cache_dir.joinpath(f"{cache_key}.pxsim")

    try:
        s = summarizer.from_file(cache_path)
    except FileNotFoundError:
        print(f"Cache miss for config with hash {cache_key}. Running simulation and saving summary to cache.")
        s = sim.run(summarizer=summarizer, **kwargs)
        s.to_file(cache_path)
    else:
        print(f"Cache hit for config with hash {cache_key}. Loaded summary from cache.")

    return s
