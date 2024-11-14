from __future__ import annotations

import os
import pathlib
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

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
from passengersim.core import __version__ as _passengersim_core_version

from . import MultiSimulation, Simulation
from .config import Config
from .summaries import SimulationTables

if TYPE_CHECKING:
    UseExistingT = Literal[True, False, "ignore", "raise"]


class Experiment:
    def __init__(
        self,
        title: str,
        tag: str | None = None,
        multiprocess: bool = True,
        *,
        external: str | os.PathLike | None = None,
    ):
        self.title = title
        self.tag = tag
        self.multi = multiprocess
        self.external = external
        self.func = None

    def __call__(self, func: Callable[[Config], Config] | Config):
        if self.func is None:
            # decorate a function that takes a config and returns a modified config
            self.func = func
            if self.tag is None:
                self.tag = func.__name__
            return self
        else:
            # if the function is already decorated, call it with the new function,
            # while making a deep copy of the config to avoid modifying the original
            if isinstance(func, Config):
                return self.func(func.model_copy(deep=True))
            else:
                raise TypeError(
                    "Experiment already decorated, expected base_config as input"
                )


class Experiments:
    def __init__(
        self,
        config: Config,
        output_dir: pathlib.Path | None = None,
        pickle: bool | str = "passengersim_output",
        html: bool | str = "passengersim_output",
    ):
        self.experiments: list[Experiment] = []
        self.base_config = config
        self.output_dir = output_dir
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

    def _rename_file(self, tag: str, filename: pathlib.Path):
        if self.output_dir is None:
            return pathlib.Path(tag) / filename
        else:
            return pathlib.Path(self.output_dir) / tag / pathlib.Path(filename).name

    def __call__(
        self,
        title: str,
        tag: str | None = None,
        multiprocess: bool = True,
        *,
        external: str | os.PathLike | None = None,
    ):
        e = Experiment(title, tag, multiprocess, external=external)
        self.experiments.append(e)
        return e

    @staticmethod
    def _check_loaded_summary(
        summary: SimulationTables,
        config: Config,
        tag: str,
        check_versions: bool = True,
        check_content: bool = True,
    ) -> tuple[str, SimulationTables | None]:
        """
        Check if the loaded summary matches the config and PassengerSim versions.

        Parameters
        ----------
        summary : SimulationTables
        config : Config
        tag : str
        check_versions : bool, optional
            If True, check the PassengerSim versions in the loaded summary.
        check_content : bool, optional
            If True, check the content of the loaded summary.

        Returns
        -------
        str
            A message about the loaded summary
        SimulationTables
            The loaded summary if it matches the config, otherwise None
        """
        msg = ""
        check = config.find_differences(summary.config)
        try:
            versions = summary.metadata("version")
        except KeyError:
            msg = (
                f"Loaded {tag} from {config.outputs.pickle}, "
                f"but the PassengerSim version is unknown"
            )
            return msg, None

        public_version = versions.get("passengersim", None)
        core_version = versions.get("passengersim_core", None)

        if check_versions and public_version is None:
            msg = (
                f"Loaded {tag} from {config.outputs.pickle}, "
                f"but the PassengerSim version is unknown"
            )
            return msg, None

        if check_versions and public_version != _passengersim_version:
            msg = (
                f"Loaded {tag} from {config.outputs.pickle}, "
                f"but the PassengerSim version has changed: "
                f"running {_passengersim_version}, found {public_version}"
            )
            return msg, None

        if check_versions and core_version is None:
            msg = (
                f"Loaded {tag} from {config.outputs.pickle}, "
                f"but the PassengerSim.Core version is unknown"
            )
            return msg, None

        if check_versions and core_version != _passengersim_core_version:
            msg = (
                f"Loaded {tag} from {config.outputs.pickle}, "
                f"but the PassengerSim.Core version has changed: "
                f"running {_passengersim_core_version}, found {core_version}"
            )
            return msg, None

        if check_content and check:
            msg = (
                f"Loaded {tag} from {config.outputs.pickle}, "
                f"but the config has changed:\n{check}"
            )
            return msg, None
        elif check:
            msg = (
                f"Loaded {tag} from {config.outputs.pickle}, "
                f"but the config has changed:\n{check}"
            )
            warnings.warn(msg, stacklevel=2)
            return msg, summary

        if check:
            msg = (
                f"Loaded {tag} from {config.outputs.pickle}, "
                f"although the config has changed:\n{check}"
            )
        else:
            msg = f"Loaded {tag} from {config.outputs.pickle}"

        return msg, summary

    def run(
        self,
        use_existing: UseExistingT | dict[str, UseExistingT] = True,
        *,
        tag: str | None = None,
        check_versions: bool = True,
        check_content: bool = True,
    ):
        """
        Run the experiments.

        Parameters
        ----------
        use_existing : Literal[True, False, "ignore", "raise"] or dict
            This can either be a single value for all experiments, or a dictionary
            mapping tags to values.  For each value, the behavior is as follows:
            If True, load from existing output pickle files if they exist,
            otherwise run the simulation for each experiment.  If False, always
            run the simulation for each experiment. If "ignore", load results
            from output pickle files if they exist, otherwise skip each
            experiment.  If "raise", raise an error if the output pickle files
            do not exist for any experiment.
        tag : str, optional
            If provided, only run the experiment with the given tag.
        check_versions : bool, default True
            If True, check the PassengerSim versions in the loaded summary (if
            any), and re-run the simulation if they do not match the current
            environment. If False, do not check the PassengerSim versions.
        check_content : bool, default True
            If True, check the content of the loaded summary (if any), and
            re-run the simulation if the config has changed. If False, do not
            check the content of the loaded summary.

        Returns
        -------
        contrast.Contrast or SimulationTables
        """

        results = contrast.Contrast()

        # validate that all experiments have unique tags
        tags = set()
        for e in self.experiments:
            if e.tag is None:
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
            top_task = top_progress.add_task(
                "[blue]Experiments", total=len(selected_experiments)
            )

            for e in selected_experiments:
                top_progress.update(
                    top_task, advance=1, description=f"[bold blue]{e.tag}", refresh=True
                )

                if e.external:
                    # If an external file is provided, load it and skip the simulation.
                    # This is done without regard for the use_existing parameter, and
                    # the absence of the external file is always an error.
                    summary = SimulationTables.from_pickle(e.external)
                    live_display.console.print(
                        f"Loaded experiment {e.tag} from {e.external}"
                    )
                    results[e.tag] = summary
                    continue

                # Create the modified config for this experiment
                config = e.func(self.base_config.model_copy(deep=True))
                config.outputs.html.title = e.title

                # Update the paths for the output files
                if config.outputs.html.filename:
                    config.outputs.html.filename = self._rename_file(
                        e.tag, config.outputs.html.filename
                    )
                if config.outputs.pickle:
                    config.outputs.pickle = self._rename_file(
                        e.tag, config.outputs.pickle
                    )
                if config.outputs.excel:
                    config.outputs.excel = self._rename_file(
                        e.tag, config.outputs.excel
                    )

                summary = None

                e_use_existing = use_existing.get(e.tag, default_use_existing)
                if e_use_existing:
                    # Check if the output pickle files already exist
                    try:
                        summary = SimulationTables.from_pickle(config.outputs.pickle)
                    except FileNotFoundError:
                        if e_use_existing == "raise":
                            raise
                        elif e_use_existing == "ignore":
                            continue
                    else:
                        # If we reach this point, we have successfully loaded the
                        # output pickle file. But before we congratulate ourselves,
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
                                raise
                            elif e_use_existing == "ignore":
                                continue

                if summary is None:
                    # If we reach this point, we need to run the simulation

                    # Initialize the simulation
                    if e.multi:
                        sim = MultiSimulation(config)
                        summary = sim.run(rich_progress=rich_progress)
                        del sim
                    else:
                        sim = Simulation(config)
                        summary = sim.run()
                        del sim

                results[e.tag] = summary

            top_progress.update(
                top_task,
                description="[bold blue]Finished Experiments",
                refresh=True,
                visible=False,
            )

        if tag is not None and len(selected_experiments) == 1:
            return results[selected_experiments[0].tag]
        return results
