from __future__ import annotations

import hashlib
import pathlib
from typing import TYPE_CHECKING, Any, TypeVar

from passengersim import Config, MultiSimulation, Simulation
from passengersim.summaries.generic import GenericSimulationTables

if TYPE_CHECKING:
    from pydantic.main import IncEx

SimulationTablesT = TypeVar("SimulationTablesT", bound=GenericSimulationTables)


def generate_hash(
    cfg: Config, *, include: IncEx | None = None, exclude: IncEx | None = None, extra: list[str] | None = None
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
    **kwargs: Any,
) -> SimulationTablesT:
    """Reuse saved summaries without requiring or restoring a detail database.

    Parameters
    ----------
    sim : Simulation or MultiSimulation
        Simulation to run on a cache miss.
    cache_dir : str or pathlib.Path
        Directory containing summary archives keyed by configuration.
    summarizer : type or GenericSimulationTables, optional
        Summary implementation. Defaults to the standard simulation summarizer.
    key : str, default ""
        Explicit cache key; otherwise derive it from the configuration.
    key_check : str, default ""
        Expected configuration hash; reject a mismatch before using the cache.
    **kwargs
        Arguments forwarded to the simulation on a cache miss.

    Returns
    -------
    GenericSimulationTables
        Saved summaries on a hit, or the live result on a miss. Only the live
        result may retain a detail database.

    Raises
    ------
    ValueError
        The expected hash differs or the simulation uses a caller-owned connection.
    """

    cfg: Config = sim.config

    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if summarizer is None:
        from passengersim.driver import get_default_summarizer

        summarizer = get_default_summarizer()

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

    from passengersim.driver._trial_database import finalize_result

    if getattr(sim, "_borrowed_connection", None) is not None:
        raise ValueError(
            "cache_dir cannot populate a caller-owned connection on a cache hit; "
            "run without caching and export explicitly"
        )

    try:
        s = summarizer.from_file(cache_path, read_latest=False)
    except FileNotFoundError:
        print(f"Cache miss for config with hash {cache_key}. Running simulation and saving summary to cache.")
        s = sim.run(summarizer=summarizer, **kwargs)
        s.to_file(cache_path, add_timestamp_ext=False)
    else:
        print(f"Cache hit for config with hash {cache_key}. Loaded summary from cache.")

    return finalize_result(s)
