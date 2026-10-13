"""Resolve storage policy before allocating files or starting workers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from passengersim.config import Config
    from passengersim.summaries.generic import GenericSimulationTables

from .database import DatabaseConfig


@dataclass(frozen=True)
class StoragePolicy:
    """Resolved intent; runtime paths are allocated separately by their owner."""

    mode: Literal["disabled", "borrowed", "temporary", "persistent", "memory", "summary_only"]
    filename: Path | None = None
    directory: Path | None = None


def resolve_storage(db: DatabaseConfig | None, *, orchestrated=False, borrowed=False) -> StoragePolicy:
    """Validate combinations without allocating files or mutating configuration."""
    if borrowed:
        if db is None or db.engine is None:
            raise ValueError("A supplied connection conflicts with disabled database recording")
        if db.filename is not None or db.result_storage is not None or db.result_directory is not None:
            raise ValueError("A supplied connection cannot be combined with managed storage settings")
        return StoragePolicy("borrowed")
    if db is None or db.engine is None:
        return StoragePolicy("disabled")
    memory_alias = str(db.filename) == ":memory:"
    mode = db.result_storage or ("memory" if memory_alias else "persistent" if db.filename is not None else "temporary")
    filename = None if memory_alias else db.filename
    if memory_alias and mode not in {"memory", "summary_only"}:
        raise ValueError("filename=':memory:' conflicts with the selected storage policy")
    if filename is not None and db.result_directory is not None:
        raise ValueError("Specify either db.filename or db.result_directory, not both")
    if mode != "persistent" and filename is not None:
        raise ValueError(f"{mode} storage cannot have a persistent filename; remove db.filename")
    if mode in {"memory", "summary_only"} and db.result_directory is not None:
        raise ValueError(f"{mode} storage cannot have a result_directory")
    if mode == "persistent" and filename is None and db.result_directory is None:
        raise ValueError("Persistent storage requires db.filename or db.result_directory")
    if mode == "memory" and orchestrated:
        raise ValueError(
            "Memory storage is standalone-only; use temporary or summary_only for MultiSimulation/Experiments"
        )
    return StoragePolicy(
        mode,
        Path(filename).absolute() if filename is not None else None,
        Path(db.result_directory).absolute() if db.result_directory is not None else None,
    )


def validate_database_reports(
    config: Config,
    policy: StoragePolicy,
    summarizer: type[GenericSimulationTables] | GenericSimulationTables | None = None,
) -> None:
    """Reject requested SQL-backed reports when recording is disabled.

    Parameters
    ----------
    config : Config
        Simulation configuration containing the requested reports.
    policy : StoragePolicy
        Resolved database recording and retention policy.
    summarizer : type or GenericSimulationTables, optional
        Summary implementation defining the SQL-backed reports. Defaults to
        SimulationTables.

    Raises
    ------
    ValueError
        Recording is disabled and at least one SQL-backed report is requested.
    """
    if policy.mode != "disabled":
        return
    if summarizer is None:
        from passengersim.summaries import SimulationTables

        summarizer = SimulationTables
    requested = {item if isinstance(item, str) else item[0] for item in config.outputs.reports}
    queries = set(summarizer._std_query)
    if requested & queries or "*" in requested:
        raise ValueError("Database recording is disabled; remove database-dependent reports or enable db.engine")
