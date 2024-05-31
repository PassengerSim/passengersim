import pathlib

from .pretty import PrettyModel


class OutputConfig(PrettyModel, extra="forbid", validate_assignment=True):
    log_reports: bool = False
    """Write basic reports directly to the run log."""

    excel: pathlib.Path | None = None
    """Write excel outputs to this file after a run."""

    reports: set[str | tuple[str, ...]] = {
        "fare_class_mix",
        "load_factors",
        "bookings_by_timeframe",
        "total_demand",
        "load_factor_distribution",
    }
    """Reports to include."""

    # TODO what reports require what database items?
    # e.g. demand_to_come requires we store all `demand` not just demand_final
