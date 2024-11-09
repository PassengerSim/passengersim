import pathlib

from pydantic import field_serializer

from .pretty import PrettyModel


class HtmlOutputConfig(PrettyModel, extra="forbid", validate_assignment=True):
    """Configuration for HTML outputs."""

    filename: pathlib.Path | None = None
    """Write HTML outputs to this file after a run.

    If this is None, no HTML outputs will be written."""

    title: str | None = None
    """Title of the HTML report.

    If this is None, the title will be the scenario name."""

    carrier_revenues: bool = True
    """Include carrier revenues in the HTML report."""

    carrier_load_factors: bool = True
    """Include carrier load factors in the HTML report."""

    carrier_yields: bool = True
    """Include carrier yields in the HTML report."""

    fare_class_mix: bool = True
    """Include fare class mix in the HTML report.

    This figure is always by carrier."""

    bookings_by_timeframe: bool | list[str] = True
    """Include bookings by timeframe in the HTML report.

    If this is a list of strings, include only the specified carriers."""

    leg_load_factor_distribution: bool = True
    """Include leg load factor distribution in the HTML report."""

    bid_price_history: bool = True
    """Include bid price history in the HTML report."""

    displacement_history: bool = True
    """Include displacement history in the HTML report."""

    carrier_table: bool = True
    """Include carrier table in the HTML report."""

    other: list[str | tuple[str, dict]] = []

    configs: list[str] = [
        "carriers",
        "rm_systems",
        "simulation_controls",
        "db",
        "outputs",
    ]
    """Include these configuration items in the HTML report."""

    def __bool__(self) -> bool:
        return self.filename is not None


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

    If this is a set containing only "*", all reports will be included; this
    may be computationally expensive.
    """

    html: HtmlOutputConfig = HtmlOutputConfig()
    """Configuration for HTML outputs."""

    pickle: pathlib.Path | None = None
    """Write a pickle of the SimulationTables output to this file after a run."""

    # TODO what reports require what database items?
    # e.g. demand_to_come requires we store all `demand` not just demand_final

    @field_serializer("reports", when_used="json")
    def serialize_reports(self, reports: set):
        return list(reports)
