import pathlib

from .pretty import PrettyModel


class OutputConfig(PrettyModel, extra="forbid", validate_assignment=True):
    log_reports: bool = False
    """Write basic reports directly to the run log."""

    excel: pathlib.Path | None = None
    """Write excel outputs to this file after a run."""

    reports: set[str | tuple[str, ...]] = set()
    """Reports to include.

    Database queries reports can be included here.  This is important for
    multiprocessing runs with in-memory databases, as database results will not
    be available after the database connection is closed in each subprocess.
    """

    # TODO what reports require what database items?
    # e.g. demand_to_come requires we store all `demand` not just demand_final
