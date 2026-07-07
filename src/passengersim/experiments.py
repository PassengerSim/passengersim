"""Run structured sets of PassengerSim simulations as experiments.

Supports defining parameter changes and scenario comparisons, running
individual simulations sequentially or in parallel across multiple
processes, and collecting results into a unified output.
"""

from __future__ import annotations

import concurrent.futures
import pathlib
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from passengersim import __version__ as _passengersim_version
from passengersim import contrast
from passengersim.callbacks import CallbackMixin
from passengersim.core import __version__ as _passengersim_core_version

from . import MultiSimulation, Simulation
from ._types import PathLike
from .config import Config
from .driver import check_summarizer, get_default_summarizer
from .mp_executor import JobExecutor
from .summaries import GenericSimulationTables, SimulationTables

if TYPE_CHECKING:
    from passengersim.contrast import Contrast

    UseExistingT = Literal[True, False, "ignore", "raise"]


class OverwriteExperimentWarning(UserWarning):
    """Warning raised when an experiment is overwritten by another with the same tag."""


class Experiment:
    """A single experiment definition and its associated config-transforming callable.

    Instances are normally created via the :class:`Experiments` factory methods or
    by decorating a function with an :class:`Experiment` instance.  The instance
    stores the experiment metadata (title, tag, multiprocess flag), the
    config-transforming function, and optionally pre-loaded or cached results.
    """

    def __init__(
        self,
        title: str | None,
        tag: str | None = None,
        multiprocess: bool = True,
        *,
        external: GenericSimulationTables | PathLike | None = None,
    ) -> None:
        """Initialize an experiment definition.

        Parameters
        ----------
        title : str or None
            Human-friendly title for the experiment. Used in HTML reports. If
            None, the tag is used as the title.
        tag : str, optional
            A short machine-friendly tag for the experiment. Ideally this tag will not have
            spaces or other special character other than underscores. This tag is used as a key
            in the results dictionary returned by `Experiments.run()`, and it is used in
            generating filenames for the experiment outputs.  If not provided, the name of
            the decorated function is used as the tag.
        multiprocess : bool, default True
            If True, run the simulation for this experiment in multi-process mode.  If False,
            run the simulation for this experiment in single-process mode.  Note that
            multi-process mode is not compatible with all environments, and may cause issues
            in interactive environments such as Jupyter notebooks.  In those cases, set this to
            False to run in single-process mode.
        external : GenericSimulationTables or path-like, optional
            If provided, this should be an existing SimulationTables result or a path to an
            existing output file containing the results for this experiment.  If this is
            provided, the experiment will skip running the simulation and instead load the
            results from the given file.  This is useful for cases where the simulation has
            already been run and the results are saved, but you want to include those results
            in a report with other experiments, without re-running the simulation.  If given
            a path but the given file does not exist or cannot be loaded, an error will
            be raised.
        """
        self.title = title
        self.tag = tag
        self.multi = multiprocess
        self.external = external
        self.func = None
        if isinstance(self.external, SimulationTables):
            self.cached = self.external
            self.external = None
        else:
            self.cached = None

    def __call__(self, func: Callable[[Config], Config] | Config) -> Experiment | Config:
        """Decorate a config-mutating function or execute the experiment.

        When called for the first time with a callable, this method stores the
        function and returns ``self`` so the instance can be used as a
        decorator.  When called again with a :class:`Config`, it runs the
        stored function against a deep copy of that config and returns the
        modified config.

        Parameters
        ----------
        func : callable or Config
            The function to decorate, or a base :class:`Config` to run the
            experiment against.

        Returns
        -------
        Experiment or Config
            ``self`` when decorating a function; the modified config when
            invoking the experiment.

        Raises
        ------
        TypeError
            If the experiment has already been decorated and the argument is
            not a :class:`Config`.
        """
        if self.func is None:
            # decorate a function that takes a config and returns a modified config
            self.func = func
            if self.tag is None:
                self.tag = func.__name__
            if self.title == "__DEFERRED_INIT__":
                self.title = self.tag
            return self
        else:
            # if the function is already decorated, call it with the base config,
            # while making a deep copy to avoid modifying the original
            if isinstance(func, Config) or type(func).__name__ == "Config":
                return self.func(func.model_copy(deep=True))
            else:
                raise TypeError("Experiment already decorated, expected base_config as input")


class Experiments(CallbackMixin):
    """Manage and run a collection of PassengerSim experiments.

    This class stores a base :class:`Config`, creates per-experiment
    configurations, orchestrates sequential or parallel execution, and
    collects the results into a :class:`~passengersim.contrast.Contrast`
    dictionary.  It also manages output paths, report generation, and
    optional result caching.
    """

    _report_filename = None

    def __init__(
        self,
        config: Config,
        output_dir: pathlib.Path | None | Literal[False] = None,
        *,
        pickle: bool | str = False,
        html: bool | str = "passengersim_output",
        hide_from_git: bool = True,
    ) -> None:
        """Initialize the experiment manager.

        Parameters
        ----------
        config : Config
            Base configuration used to create per-experiment deep copies.
        output_dir : pathlib.Path or None or False, optional
            Directory where all experiment outputs are written. If ``None``
            (default), outputs are placed in tag-named subdirectories of the
            current directory. If ``False``, output is disabled.
        pickle : bool or str, default False
            If truthy, ensure the base config is set up to write pickle output.
            If ``True``, the default filename stem ``"passengersim_output"``
            is used.
        html : bool or str, default "passengersim_output"
            If truthy, ensure the base config is set up to write HTML output.
            If ``True``, the default filename stem ``"passengersim_output"``
            is used.
        hide_from_git : bool, default True
            If True and ``output_dir`` is set, write a ``.gitignore`` file
            that prevents git from tracking generated outputs.

        Returns
        -------
        None
        """
        self.experiments: list[Experiment] = []
        self.base_config = config
        self.output_dir = output_dir
        if isinstance(self.output_dir, str):
            self.output_dir = pathlib.Path(output_dir)
        self.extra_reporting = None
        if self.output_dir and hide_from_git:
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True)
            if not self.output_dir.joinpath(".gitignore").exists():
                self.output_dir.joinpath(".gitignore").write_text("**\n")
        # ensure the base config has pickle output
        if pickle and self.base_config.outputs.pickle is None:
            if pickle is True:
                pickle = "passengersim_output"
            self.base_config.outputs.pickle = pathlib.Path(pickle)
        # ensure the base config has html output
        if html and self.base_config.outputs.html.filename is None:
            if html is True:
                html = "passengersim_output"
            self.base_config.outputs.html.filename = pathlib.Path(html)

        self._sims = None
        # sims are only retained if requested

    @property
    def sims(self) -> dict[str, Simulation | MultiSimulation]:
        """Retained simulation objects from the most recent run.

        Returns
        -------
        dict[str, Simulation or MultiSimulation]
            Mapping from experiment tag to the simulation instance that
            produced the corresponding results.

        Raises
        ------
        ValueError
            If ``retain_sims=True`` was not passed to :meth:`run`.
        """
        if self._sims is None:
            raise ValueError("sims not available; set retain_sims=True in run() to retain them")
        return self._sims

    def _rename_file(self, tag: str, filename: pathlib.Path | str | None) -> pathlib.Path | None:
        """Rewrite an output filename so it lives under an experiment-tag subdirectory.

        Parameters
        ----------
        tag : str
            Experiment tag used as the subdirectory name.
        filename : pathlib.Path or str or None
            Original output filename to relocate.

        Returns
        -------
        pathlib.Path or None
            The rewritten path rooted under ``tag``. Returns the original
            value unchanged for non-path types and ``None`` when output is
            disabled.
        """
        if not isinstance(filename, str | pathlib.Path):
            return filename
        if self.output_dir is None:
            return pathlib.Path(tag) / filename
        elif not self.output_dir:
            return None
        else:
            return pathlib.Path(self.output_dir) / tag / pathlib.Path(filename).name

    def existing(self, external: GenericSimulationTables | PathLike | None = None) -> Experiment:
        """Create an experiment that uses existing results, rather than running a simulation.

        Parameters
        ----------
        external : GenericSimulationTables or path-like, optional
            If provided, this should be an existing SimulationTables result or a path to an
            existing output file containing the results for this experiment.  If this is
            provided, the experiment will skip running the simulation and instead load the
            results from the given file.  This is useful for cases where the simulation has
            already been run and the results are saved, but you want to include those results
            in a report with other experiments, without re-running the simulation.  If given
            a path but the given file does not exist or cannot be loaded, an error will
            be raised.

        Returns
        -------
        Experiment
            An experiment that will use the given existing results when run, rather than running
            a simulation.
        """
        return self(external=external)

    def __call__(
        self,
        title: str | Callable[[Config], Config] = "__DEFERRED_INIT__",
        tag: str | None = None,
        multiprocess: bool = True,
        *,
        external: GenericSimulationTables | PathLike | None = None,
    ) -> Experiment:
        """Create a new experiment or register a decorated function as one.

        Can be called directly to create a titled experiment, used as a plain
        decorator (``@exps``), or used as a parameterised decorator
        (``@exps("Title")``).  Duplicate tags cause the old experiment to be
        replaced with a warning.

        Parameters
        ----------
        title : str or callable, default "__DEFERRED_INIT__"
            Human-friendly title for the experiment, or the function to
            decorate when this method is used directly as a decorator.
        tag : str, optional
            Machine-friendly tag. Defaults to the decorated function name.
        multiprocess : bool, default True
            If True, allow multi-process execution for this experiment.
        external : GenericSimulationTables or path-like, optional
            Existing results to use instead of running the simulation.

        Returns
        -------
        Experiment
            The newly created (or updated) experiment object.
        """
        if title == "__DEFERRED_INIT__":
            e = Experiment(title, tag, multiprocess, external=external)
        elif not isinstance(title, str):
            # called as a decorator, so the first argument is the function
            e = Experiment(None, tag, multiprocess, external=external)(title)
        else:
            e = Experiment(title, tag, multiprocess, external=external)
        # check if this is a duplicate of an existing experiment
        # if so, overwrite the existing experiment and warn the user
        for i in range(len(self.experiments)):
            if e.tag == self.experiments[i].tag:
                warnings.warn(
                    f"Overwriting existing experiment tag: {e.tag}", stacklevel=2, category=OverwriteExperimentWarning
                )
                self.experiments[i] = e
                return e
        # otherwise add the new experiment
        self.experiments.append(e)
        return e

    @staticmethod
    def _check_loaded_summary(
        summary: GenericSimulationTables,
        config: Config,
        tag: str,
        check_versions: bool = True,
        check_content: bool = True,
        source_file: str | None = None,
    ) -> tuple[str, GenericSimulationTables | None]:
        """Check whether a loaded summary matches the expected config and versions.

        Parameters
        ----------
        summary : GenericSimulationTables
            Loaded summary object to validate.
        config : Config
            Expected configuration to compare against the summary's stored
            config.
        tag : str
            Experiment tag used in diagnostic messages.
        check_versions : bool, default True
            If True, verify that the `passengersim` and `passengersim.core`
            package versions recorded in the summary match the currently
            installed versions.
        check_content : bool, default True
            If True, treat any config differences as a mismatch and return
            ``None`` for the summary.
        source_file : str, optional
            Source path to include in diagnostic messages. If omitted, the
            method attempts to infer it from the summary metadata or config.

        Returns
        -------
        tuple[str, GenericSimulationTables or None]
            A two-element tuple: a human-readable status message and the
            summary itself if it passes all checks, or ``None`` if it should
            not be reused.
        """
        if source_file is None:
            try:
                source_file = summary.metadata("store.filename")
            except (KeyError, Exception):
                pass
        if source_file is None:
            source_file = config.outputs.pickle
        if source_file is None:
            source_file = config.outputs.disk
        msg = ""
        try:
            check = config.find_differences(summary.config)
        except ValidationError as e:
            check = e
        try:
            versions = summary.metadata("version")
        except KeyError:
            msg = f"Loaded {tag} from {source_file}, but the PassengerSim version is unknown"
            return msg, None

        public_version = versions.get("passengersim", None)
        core_version = versions.get("passengersim_core", None)

        if check_versions and public_version is None:
            msg = f"Loaded {tag} from {source_file}, but the PassengerSim version is unknown"
            return msg, None

        if check_versions and public_version != _passengersim_version:
            msg = (
                f"Loaded {tag} from {source_file}, "
                f"but the PassengerSim version has changed: "
                f"running {_passengersim_version}, found {public_version}"
            )
            return msg, None

        if check_versions and core_version is None:
            msg = f"Loaded {tag} from {source_file}, but the PassengerSim.Core version is unknown"
            return msg, None

        if check_versions and core_version != _passengersim_core_version:
            msg = (
                f"Loaded {tag} from {source_file}, "
                f"but the PassengerSim.Core version has changed: "
                f"running {_passengersim_core_version}, found {core_version}"
            )
            return msg, None

        if isinstance(check, ValidationError):
            msg = f"Loaded {tag} from {source_file}, but the config is invalid:\n{str(check)[:4000]}"
            return msg, None

        if check_content and check:
            msg = f"Loaded {tag} from {source_file}, but the config has changed:\n{str(check)[:4000]}"
            return msg, None

        if check:
            msg = f"Loaded {tag} from {source_file}, although the config has changed:\n{str(check)[:4000]}"
        else:
            msg = f"Loaded {tag} from {source_file}"

        return msg, summary

    def _write_report_after_run(self, write_report: PathLike | bool | None, results: Contrast) -> None:
        """Write the HTML report after all experiments have finished.

        Parameters
        ----------
        write_report : path-like or bool or None
            Destination control flag as originally passed to :meth:`run`. A
            path-like value is used directly; any other truthy value uses the
            default filename ``"experiments-summary.html"``.
        results : Contrast
            The collected experiment results to include in the report.

        Returns
        -------
        None
        """
        if isinstance(write_report, PathLike):
            write_report = pathlib.Path(write_report)
        else:
            write_report = pathlib.Path("experiments-summary.html")
        # if output directory is set, write the report there,
        # unless the path is absolute (then write it to the given path)
        if self.output_dir and not write_report.is_absolute():
            write_report = self.output_dir / write_report
        self._report_filename = results.write_report(
            write_report, base_config=self.base_config, extra=self.extra_reporting
        )

    def _run_experiments_in_sequence(
        self,
        use_existing: UseExistingT | dict[str, UseExistingT] = True,
        *,
        tag: str | None = None,
        check_versions: bool = True,
        check_content: bool = True,
        single_process: bool = False,
        retain_sims: bool = False,
        write_report: PathLike | bool | None = True,
        cache_results: bool = True,
    ) -> contrast.Contrast | GenericSimulationTables:
        """Run the selected experiments sequentially in the calling process.

        Parameters
        ----------
        use_existing : Literal[True, False, "ignore", "raise"] or dict, default True
            Single value applied to all experiments, or a mapping from
            experiment tag to a per-experiment value.

            - ``True`` – load from an existing output file if one is found,
              otherwise run the simulation.
            - ``False`` – always run the simulation.
            - ``"ignore"`` – load from an existing output file if found,
              otherwise skip the experiment entirely.
            - ``"raise"`` – raise an error if the output file is missing.
        tag : str, optional
            If provided, run only the experiment with this tag and return its
            result directly rather than a :class:`~passengersim.contrast.Contrast`.
        check_versions : bool, default True
            If True, re-run the simulation when a loaded summary was produced
            by a different PassengerSim version.
        check_content : bool, default True
            If True, re-run the simulation when a loaded summary's config
            differs from the current config.
        single_process : bool, default False
            If True, force all experiments to run in single-process mode,
            overriding each experiment's ``multiprocess`` flag.
        retain_sims : bool, default False
            If True, keep simulation objects in :attr:`sims` after completion.
            Primarily useful for debugging.
        write_report : path-like or bool or None, default True
            If truthy, write an HTML report when all experiments finish. Pass
            a path-like value to specify the destination; ``True`` uses the
            default filename ``"experiments-summary.html"``.
        cache_results : bool, default True
            If True, cache each experiment's result on the corresponding
            :class:`Experiment` object so it can be reused without reloading.

        Returns
        -------
        contrast.Contrast or GenericSimulationTables
            A :class:`~passengersim.contrast.Contrast` mapping tags to results,
            or a single result when ``tag`` selects exactly one experiment.
        """

        results = contrast.Contrast()

        if retain_sims:
            self._sims = {}

        # validate that all experiments have unique tags
        tags = set()
        for e in self.experiments:
            if e.tag is None:
                if e.title is None:
                    raise ValueError("Experiment missing tag and title")
                raise ValueError("Experiment missing tag: " + e.title)
            if e.tag in tags:
                raise ValueError("Duplicate experiment tag: " + e.tag)
            tags.add(e.tag)

        if isinstance(tag, str):
            selected_experiments = [e for e in self.experiments if e.tag == tag]
            if not selected_experiments:
                raise ValueError(f"No experiment found with tag {tag}")
        elif tag is None:
            selected_experiments = self.experiments
        else:
            raise TypeError("tag must be a string or None")

        rich_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            auto_refresh=False,
            transient=True,
        )
        top_progress = Progress(
            MofNCompleteColumn(),
            TextColumn("[progress.description]{task.description}"),
            auto_refresh=False,
            transient=True,
        )
        live_display = Live(
            Panel(
                Group(
                    top_progress,
                    rich_progress,
                ),
                title="Experiments",
                border_style="blue",
                expand=True,
            ),
            refresh_per_second=4,
            transient=True,
        )

        default_use_existing = True
        if not isinstance(use_existing, dict):
            default_use_existing = use_existing
            use_existing = {}

        with live_display:
            top_task = top_progress.add_task("[blue]Experiments", total=len(selected_experiments))

            for e in selected_experiments:
                top_progress.update(top_task, advance=1, description=f"[bold blue]{e.tag}", refresh=True)

                if e.cached:
                    # If a cached SimulationTables result is available, use it and skip the simulation.
                    # No checks are performed on the cached result, so it is the user's responsibility to
                    # ensure that the cached result is valid and matches the current config and PassengerSim
                    # versions, if applicable.
                    summary = e.cached
                    live_display.console.print(f"Using cached results for experiment {e.tag}")
                    results[e.tag] = summary
                    continue
                elif e.external:
                    # If an external file is provided, load it and skip the simulation.
                    # This is done without regard for the use_existing parameter, and
                    # the absence of the external file is always an error.
                    if isinstance(e.external, GenericSimulationTables):
                        summary = e.external
                    else:
                        summary = get_default_summarizer().from_file(e.external)
                    live_display.console.print(f"Loaded experiment {e.tag} from {e.external}")
                    results[e.tag] = summary
                    continue

                # Create the modified config for this experiment
                config = e.func(self.base_config.model_copy(deep=True))
                config.outputs.html.title = e.title or e.tag

                # Update the paths for the output files
                if config.outputs.html.filename:
                    config.outputs.html.filename = self._rename_file(e.tag, config.outputs.html.filename)
                if config.outputs.pickle:
                    config.outputs.pickle = self._rename_file(e.tag, config.outputs.pickle)
                if config.outputs.excel:
                    config.outputs.excel = self._rename_file(e.tag, config.outputs.excel)

                summary = None

                e_use_existing = use_existing.get(e.tag, default_use_existing)
                if e_use_existing:
                    try:
                        # Check if the output pickle files are defined and already exist
                        if config.outputs.pickle:
                            summary = get_default_summarizer().from_pickle(config.outputs.pickle)
                        else:
                            raise FileNotFoundError("No output pickle file specified")
                    except FileNotFoundError:
                        # At this point either the output pickle file is not defined
                        # (triggering the explicit FileNotFoundError) or does not exist,
                        # which raises the FileNotFoundError organically.  Either way,
                        # we also want to check if the pxsim-format disk file exists.

                        try:
                            second_file = config.outputs._get_disk_filename()
                            if second_file:
                                summary = get_default_summarizer().from_file(second_file)
                            else:
                                raise FileNotFoundError("No output disk file specified")
                        except FileNotFoundError as second_error:
                            if e_use_existing == "raise":
                                # Neither the output pickle file nor the pxsim-format disk file
                                # exist, so we need to raise an error.
                                raise second_error
                            elif e_use_existing == "ignore":
                                # Neither the output pickle file nor the pxsim-format disk file
                                # exist, but we have been instructed to ignore this.  The
                                # matching simulation will not be run, and will not be included
                                # in the results.
                                continue

                    if summary is not None:
                        # If we reach this point, we have successfully loaded the
                        # output from a pickle or pxsim file. But before we celebrate,
                        # we need to make sure the run we loaded matches the config
                        # we would otherwise run, and the versions of PassengerSim
                        # match between the run and the current environment.
                        msg, summary = self._check_loaded_summary(
                            summary,
                            config,
                            e.tag,
                            check_versions=check_versions,
                            check_content=check_content,
                        )
                        live_display.console.print(msg)
                        if summary is None:
                            if e_use_existing == "raise":
                                raise ValueError("existing result does not match requested experiment")
                            elif e_use_existing == "ignore":
                                continue

                if summary is None:
                    # If we reach this point, we need to run the simulation

                    # Initialize the simulation
                    if e.multi and not single_process:
                        sim = MultiSimulation(config)
                        if retain_sims:
                            self._sims[e.tag] = sim
                        self.apply_callback_functions(sim)
                        summary = sim.run(rich_progress=rich_progress)
                        del sim
                    else:
                        sim = Simulation(config)
                        if retain_sims:
                            self._sims[e.tag] = sim
                        self.apply_callback_functions(sim)
                        summary = sim.run(rich_progress=rich_progress)
                        del sim

                results[e.tag] = summary
                if cache_results:
                    e.cached = summary

            top_progress.update(
                top_task,
                description="[bold blue]Finished Experiments",
                refresh=True,
                visible=False,
            )

        if write_report:
            self._write_report_after_run(write_report, results)

        if tag is not None and len(selected_experiments) == 1:
            return results[selected_experiments[0].tag]
        return results

    def _run_together(
        self,
        use_existing: UseExistingT | dict[str, UseExistingT] = True,
        *,
        tag: str | None = None,
        check_versions: bool = True,
        check_content: bool = True,
        retain_sims: bool = False,
        write_report: PathLike | bool | None = True,
        cache_results: bool = True,
        summarizer: type | None = None,
    ) -> contrast.Contrast | GenericSimulationTables:
        """Run the selected experiments using the parallel job executor.

        Experiments are dispatched asynchronously to a :class:`~passengersim.mp_executor.JobExecutor`
        and their futures are awaited before the method returns.

        Parameters
        ----------
        use_existing : Literal[True, False, "ignore", "raise"] or dict, default True
            Single value applied to all experiments, or a mapping from
            experiment tag to a per-experiment value.

            - ``True`` – load from an existing output file if one is found,
              otherwise run the simulation.
            - ``False`` – always run the simulation.
            - ``"ignore"`` – load from an existing output file if found,
              otherwise skip the experiment entirely.
            - ``"raise"`` – raise an error if the output file is missing.
        tag : str, optional
            If provided, run only the experiment with this tag and return its
            result directly rather than a :class:`~passengersim.contrast.Contrast`.
        check_versions : bool, default True
            If True, re-run the simulation when a loaded summary was produced
            by a different PassengerSim version.
        check_content : bool, default True
            If True, re-run the simulation when a loaded summary's config
            differs from the current config.
        retain_sims : bool, default False
            If True, keep simulation objects in :attr:`sims` after completion.
            Primarily useful for debugging.
        write_report : path-like or bool or None, default True
            If truthy, write an HTML report when all experiments finish.
        cache_results : bool, default True
            If True, cache each experiment's result on the corresponding
            :class:`Experiment` object.
        summarizer : type, optional
            Concrete summarizer class to use when dispatching simulations
            asynchronously. If None, the default summarizer is used.

        Returns
        -------
        contrast.Contrast or GenericSimulationTables
            A :class:`~passengersim.contrast.Contrast` mapping tags to results,
            or a single result when ``tag`` selects exactly one experiment.
        """
        jobber = JobExecutor().start()
        results = contrast.Contrast()
        pending_results: dict[str, concurrent.futures.Future] = {}

        summarizer = check_summarizer(summarizer)

        if retain_sims:
            self._sims = {}

        # validate that all experiments have unique tags
        tags = set()
        for e in self.experiments:
            if e.tag is None:
                if e.title is None:
                    raise ValueError("Experiment missing tag and title")
                raise ValueError("Experiment missing tag: " + e.title)
            if e.tag in tags:
                raise ValueError("Duplicate experiment tag: " + e.tag)
            tags.add(e.tag)

        if isinstance(tag, str):
            selected_experiments = [e for e in self.experiments if e.tag == tag]
            if not selected_experiments:
                raise ValueError(f"No experiment found with tag {tag}")
        elif tag is None:
            selected_experiments = self.experiments
        else:
            raise TypeError("tag must be a string or None")

        default_use_existing = True
        if not isinstance(use_existing, dict):
            default_use_existing = use_existing
            use_existing = {}

        for e in selected_experiments:
            if e.cached:
                # If a cached SimulationTables result is available, use it and skip the simulation.
                # No checks are performed on the cached result, so it is the user's responsibility to
                # ensure that the cached result is valid and matches the current config and PassengerSim
                # versions, if applicable.
                summary = e.cached
                jobber.rich_progress.console.print(f"Using cached results for experiment {e.tag}")
                results[e.tag] = summary
                continue
            elif e.external:
                # If an external file is provided, load it and skip the simulation.
                # This is done without regard for the use_existing parameter, and
                # the absence of the external file is always an error.
                if isinstance(e.external, GenericSimulationTables):
                    summary = e.external
                else:
                    summary = get_default_summarizer().from_file(e.external)
                jobber.rich_progress.console.print(f"Loaded experiment {e.tag} from {e.external}")
                results[e.tag] = summary
                continue

            # Create the modified config for this experiment
            config = e.func(self.base_config.model_copy(deep=True))
            # Revalidate the config now, which will ensure that all the changes that occur during validation
            # are captured. For example, the Experiment function might change a carrier to assign a standard
            # Frat5 curve, but that still needs to be loaded.
            config = config.model_validate(config)
            # make the config's title consistent with the experiment title or tag
            config.outputs.html.title = e.title or e.tag

            # Update the paths for the output files
            if not self.output_dir:
                config.outputs.base_dir = pathlib.Path(e.tag)
            else:
                config.outputs.base_dir = pathlib.Path(self.output_dir) / e.tag
            config.outputs.filename_stem = e.tag

            # TODO: FIND ALL THESE AND CHANGE TO config._resolve
            # if config.outputs.html.filename:
            #     config.outputs.html.filename = self._rename_file(e.tag, config.outputs.html.filename)
            # if config.outputs.pickle:
            #     config.outputs.pickle = self._rename_file(e.tag, config.outputs.pickle)
            # if config.outputs.excel:
            #     config.outputs.excel = self._rename_file(e.tag, config.outputs.excel)

            summary = None

            e_use_existing = use_existing.get(e.tag, default_use_existing)
            if e_use_existing:
                # At this point either the output pickle file is not defined
                # (triggering the explicit FileNotFoundError) or does not exist,
                # which raises the FileNotFoundError organically.  Either way,
                # we also want to check if the pxsim-format disk file exists.

                try:
                    disk_file = config.outputs.get_output_filename("disk", make_dirs=False)
                    if disk_file:
                        summary = get_default_summarizer().from_file(disk_file)
                    else:
                        raise FileNotFoundError("No output disk file specified")
                except FileNotFoundError as second_error:
                    if e_use_existing == "raise":
                        # Neither the output pickle file nor the pxsim-format disk file
                        # exist, so we need to raise an error.
                        raise second_error
                    elif e_use_existing == "ignore":
                        # The output pxsim-format disk file does not exist, but we have
                        # been instructed to ignore this.  The matching simulation will
                        # not be run, and will not be included in the results.
                        continue

                if summary is not None:
                    # If we reach this point, we have successfully loaded the
                    # output from a .pxsim file on disk. But before we celebrate,
                    # we need to make sure the run we loaded matches the config
                    # we would otherwise run, and the versions of PassengerSim
                    # match between the run and the current environment.
                    msg, summary = self._check_loaded_summary(
                        summary,
                        config,
                        e.tag,
                        check_versions=check_versions,
                        check_content=check_content,
                    )
                    jobber.rich_progress.console.print(msg)
                    if summary is None:
                        if e_use_existing == "raise":
                            raise ValueError("existing result does not match requested experiment")
                        elif e_use_existing == "ignore":
                            continue

            if summary is None:
                # If we reach this point, we need to run the simulation
                sim = MultiSimulation(config)
                if retain_sims:
                    self._sims[e.tag] = sim
                self.apply_callback_functions(sim)
                summary = sim._run_asynchronously(summarizer=summarizer, jobber=jobber, run_id=e.tag)
                del sim

            pending_results[e.tag] = summary
            results[e.tag] = summary
            if cache_results:
                e.cached = summary

        # convert cached results, which might be Futures, into finalized results
        results = contrast.Contrast(
            {k: (v.result() if isinstance(v, concurrent.futures.Future) else v) for k, v in results.items()}
        )

        # await results
        for k, v in pending_results.items():
            if isinstance(v, concurrent.futures.Future):
                results[k] = v.result()
            else:
                results[k] = v

        if write_report:
            self._write_report_after_run(write_report, results)

        if tag is not None and len(selected_experiments) == 1:
            return results[selected_experiments[0].tag]
        return results

    def run(
        self,
        use_existing: UseExistingT | dict[str, UseExistingT] = True,
        *,
        tag: str | None = None,
        check_versions: bool = True,
        check_content: bool = True,
        single_process: bool = False,
        retain_sims: bool = False,
        write_report: PathLike | bool | None = True,
        cache_results: bool = True,
    ) -> contrast.Contrast | GenericSimulationTables:
        """Run the experiments, choosing sequential or parallel execution automatically.

        When ``single_process=True`` the experiments are run in sequence via
        :meth:`_run_experiments_in_sequence`; otherwise they are dispatched to
        the parallel job executor via :meth:`_run_together`.

        Parameters
        ----------
        use_existing : Literal[True, False, "ignore", "raise"] or dict, default True
            Single value applied to all experiments, or a mapping from
            experiment tag to a per-experiment value.

            - ``True`` – load from an existing output file if one is found,
              otherwise run the simulation.
            - ``False`` – always run the simulation.
            - ``"ignore"`` – load from an existing output file if found,
              otherwise skip the experiment entirely.
            - ``"raise"`` – raise an error if the output file is missing.
        tag : str, optional
            If provided, run only the experiment with this tag and return its
            result directly rather than a :class:`~passengersim.contrast.Contrast`.
        check_versions : bool, default True
            If True, re-run the simulation when a loaded summary was produced
            by a different PassengerSim version.
        check_content : bool, default True
            If True, re-run the simulation when a loaded summary's config
            differs from the current config.
        single_process : bool, default False
            If True, force all experiments to run in single-process mode and
            execute them sequentially.
        retain_sims : bool, default False
            If True, keep simulation objects in :attr:`sims` after completion.
            Primarily useful for debugging.
        write_report : path-like or bool or None, default True
            If truthy, write an HTML report when all experiments finish. Pass
            a path-like value to specify the destination; ``True`` uses the
            default filename ``"experiments-summary.html"``.
        cache_results : bool, default True
            If True, cache each experiment's result on the corresponding
            :class:`Experiment` object so it can be reused without reloading.

        Returns
        -------
        contrast.Contrast or GenericSimulationTables
            A :class:`~passengersim.contrast.Contrast` mapping tags to results,
            or a single result when ``tag`` selects exactly one experiment.
        """
        if single_process:
            return self._run_experiments_in_sequence(
                use_existing=use_existing,
                tag=tag,
                check_versions=check_versions,
                check_content=check_content,
                single_process=single_process,
                retain_sims=retain_sims,
                write_report=write_report,
                cache_results=cache_results,
            )
        else:
            return self._run_together(
                use_existing=use_existing,
                tag=tag,
                check_versions=check_versions,
                check_content=check_content,
                retain_sims=retain_sims,
                write_report=write_report,
                cache_results=cache_results,
            )

    @property
    def report_filename(self) -> pathlib.Path:
        """Path of the HTML report written after the most recent run.

        Unless reporting is disabled, a report is written to a file after
        :meth:`run` completes. The resulting path is stored here.

        Returns
        -------
        pathlib.Path
            The path of the written report file.

        Raises
        ------
        ValueError
            If no report has been written yet.
        """
        if self._report_filename is None:
            raise ValueError("no report has been written")
        return self._report_filename

    def validate(self) -> None:
        """Validate all experiment tags and callables against the base config.

        Checks that every experiment has a unique tag and that its
        config-transforming callable runs without error when given a deep
        copy of the base config.  Does not verify that modified configs are
        mutually compatible.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any experiment is missing a tag, has a duplicate tag, or its
            callable raises an exception against the base config.
        """
        tags = set()
        for e in self.experiments:
            if e.tag is None:
                if e.title is None:
                    raise ValueError("Experiment missing tag and title")
                raise ValueError("Experiment missing tag: " + e.title)
            if e.tag in tags:
                raise ValueError("Duplicate experiment tag: " + e.tag)
            tags.add(e.tag)
            try:
                config = e.func(self.base_config.model_copy(deep=True))
                config = config.model_validate(config)
            except Exception as ex:
                raise ValueError(f"Experiment {e.tag} failed to validate") from ex
