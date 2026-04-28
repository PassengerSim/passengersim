import datetime
import pathlib
import time
from typing import Literal

from pydantic import field_serializer

from .pretty import PrettyModel


class HtmlOutputConfig(PrettyModel, extra="forbid", validate_assignment=True):
    """Configuration for HTML outputs."""

    filename: pathlib.Path | bool | None = None
    """Write HTML outputs to this file after a run.

    If this is None or False, no HTML outputs will be written. If True, HTML outputs
    will be written to a standard named file after a run, or give the exact filename
    of the file to be written."""

    title: str | None = None
    """Title of the HTML report.

    If this is None, the title will be the scenario name."""

    carrier_revenues: bool = True
    """Include carrier revenues in the HTML report."""

    carrier_total_bookings: bool = True
    """Include carrier total bookings in the HTML report."""

    carrier_revenue_distribution: bool = False
    """Include carrier revenue distribution in the HTML report."""

    carrier_load_factors: bool = True
    """Include carrier load factors in the HTML report."""

    carrier_yields: bool = True
    """Include carrier yields in the HTML report."""

    carrier_rasm: bool = True
    """Include carrier RASM in the HTML report."""

    carrier_local_share: bool = True
    """Include carrier local share in the HTML report."""

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

    segmentation_by_timeframe_table: bool = True
    """Include segmentation by timeframe table in the HTML report."""

    other: list[str | tuple[str, dict]] = []

    configs: list[str] = [
        "carriers",
        "rm_systems",
        "simulation_controls",
        "db",
        "outputs",
    ]
    """Include these configuration items in the HTML report."""

    metadata: bool = True
    """Include simulation run metadata in the HTML report."""

    def __bool__(self) -> bool:
        return bool(self.filename)


class StateOutputConfig(PrettyModel, extra="forbid", validate_assignment=True):
    save: bool = False
    """Save the state of the simulation after each trial."""

    filename: str = "[[basename]].pax-state/trial-[[trial]].pax-state"
    """Save the state of the simulation to this file after each trial.

    Accepts fields `basename` and `trial`, use double square brackets to
    insert these values.
    """

    include_trials: list[int] | None = None
    """Include trials when saving dynamic state.

    If left empty, all trials will be included.
    """

    def _resolve_filename(self, basename: str = "passengersim", trial: int = 999) -> str:
        return self.filename.replace("[[", "{").replace("]]", "}").format(basename=basename, trial=trial)


class OutputConfig(PrettyModel, extra="forbid", validate_assignment=True):
    base_dir: pathlib.Path | None = None
    """Base directory for all output files.

    Whenever an output file is designated as a simple filename or a relative path,
    it will be relative to this directory.

    If not provided, the current working directory will be used.
    """

    filename_stem: str = "passengersim-output"
    """Default stem for all output files.

    If an output file is designated as `True` instead of giving an explicit
    filename, then this stem will be used with an appropriate file extension.
    """

    log_reports: bool = False
    """Write basic reports directly to the run log."""

    excel: bool | pathlib.Path | None = None
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

    pickle: bool | pathlib.Path | None = None
    """Write a pickle of the SimulationTables output to this file after a run."""

    sim_state: StateOutputConfig = StateOutputConfig()
    """Configuration for writing simulation dynamic state after each trial.

    In contrast with the *static* state, the *dynamic* state includes things that
    are created or change during a trial of the simulation, such as attribute counters,
    histories, and summaries.  The dynamic state is the information needed to recreate
    the Simulation object (beyond the contents of the Config) to facilitate various
    debugging and introspection.
    """

    disk: bool | pathlib.Path | None = True
    """Write the SimulationTables output to this file after a run.

    This will use pxsim format, an efficient binary file that allows "lazy" file
    loading.  If set to `True`, the file will be named with the same name as
    the HTML output file, except with the extension `.pxsim`, unless there is no
    HTML output file, in which case no file will written.
    """

    def _get_disk_filename(self) -> pathlib.Path | None:
        """Get the filename for the disk output.

        If the disk output is set to `True`, this will return the filename of
        the HTML output file with the extension `.pxsim`.  If there is no HTML
        output file, this will return None.
        """
        if self.disk is True and self.html.filename is not None:
            return self.html.filename.with_suffix(".pxsim")
        elif isinstance(self.disk, pathlib.Path):
            return self.disk
        else:
            return None

    def _resolve_output_filename(
        self,
        filename: str | pathlib.Path | bool,
        suffix: str | None = None,
        timestamp: str | float | time.struct_time | datetime.datetime | None = None,
        make_dirs: bool = False,
    ) -> pathlib.Path:
        """Resolve the output filename."""
        from passengersim.utils.filenaming import filename_with_timestamp

        if filename is False:
            raise ValueError("filename is False")
        if filename is None or filename is True:
            filename = self.filename_stem
        out_dir = self.base_dir if self.base_dir is not None else pathlib.Path.cwd()
        filename: pathlib.Path = pathlib.Path(filename)
        if suffix is not None:
            suffix: str = str(suffix)
            if not suffix.startswith("."):
                suffix = f".{suffix}"
            filename = filename.with_suffix(suffix)
        if timestamp:
            filename = filename_with_timestamp(filename, timestamp=timestamp)
        if make_dirs:
            filename.parent.mkdir(parents=True, exist_ok=True)
        if filename.is_absolute():
            return filename
        else:
            return out_dir / filename

    def get_output_filename(
        self,
        which: Literal["html", "disk", "pickle", "sim_state", "excel"],
        timestamp: str | float | time.struct_time | datetime.datetime | None = None,
        make_dirs: bool = False,
        **kwargs,
    ) -> pathlib.Path | None:
        if which == "html":
            if not self.html:
                return None
            return self._resolve_output_filename(
                self.html.filename or self.filename_stem,
                suffix=".html",
                timestamp=timestamp,
                make_dirs=make_dirs,
            )
        elif which == "disk":
            if not self.disk:
                return None
            return self._resolve_output_filename(
                self.disk if isinstance(self.disk, pathlib.Path) else self.filename_stem,
                suffix=".pxsim",
                timestamp=timestamp,
                make_dirs=make_dirs,
            )
        elif which == "pickle":
            if not self.pickle:
                return None
            return self._resolve_output_filename(
                self.pickle if isinstance(self.pickle, pathlib.Path) else self.filename_stem,
                suffix=".pkl",
                timestamp=timestamp,
                make_dirs=make_dirs,
            )
        elif which == "excel":
            if not self.excel:
                return None
            return self._resolve_output_filename(
                self.excel if isinstance(self.excel, pathlib.Path) else self.filename_stem,
                suffix=".xlsx",
                timestamp=timestamp,
                make_dirs=make_dirs,
            )
        elif which == "sim_state":
            if not self.sim_state.save:
                return None
            if "basename" not in kwargs:
                kwargs["basename"] = self.filename_stem
            return self._resolve_output_filename(
                self.sim_state._resolve_filename(**kwargs),
                suffix=".pax-state",
                timestamp=timestamp,
                make_dirs=make_dirs,
            )
        else:
            raise ValueError(f"Unknown output file format: {which}")

    def _write_no_files(self):
        """Write no files."""
        self.disk = False
        self.html.filename = False
        self.pickle = False
        self.excel = False
        self.sim_state.save = False

    # TODO what reports require what database items?
    # e.g. demand_to_come requires we store all `demand` not just demand_final

    @field_serializer("reports", when_used="always")
    def serialize_reports(self, reports: set[str | tuple[str, ...]]) -> list[str | tuple[str, ...]]:
        # return a sorted list, first simple strings then tuples
        return sorted(reports, key=lambda x: (isinstance(x, tuple), x))
