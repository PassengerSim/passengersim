import os
import pathlib
from collections.abc import Callable
from typing import Literal

from passengersim import contrast

from . import MultiSimulation, Simulation
from .config import Config
from .summaries import SimulationTables


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

    def __call__(self, func: Callable[[Config], Config]):
        # decorate a function that takes a config and returns a modified config
        self.func = func
        if self.tag is None:
            self.tag = func.__name__


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

    def __call__(self, title: str, tag: str | None = None, multiprocess: bool = True):
        e = Experiment(title, tag, multiprocess)
        self.experiments.append(e)
        return e

    def run(self, use_existing: Literal[True, False, "ignore", "raise"] = True):
        """
        Run the experiments.

        Parameters
        ----------
        use_existing : Literal[True, False, "ignore", "raise"]
            If True, load from existing output pickle files if they exist,
            otherwise run the simulation for each experiment.  If False, always
            run the simulation for each experiment. If "ignore", load results
            from output pickle files if they exist, otherwise skip each
            experiment.  If "raise", raise an error if the output pickle files
            do not exist for any experiment.

        Returns
        -------
        contrast.Contrast
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

        for e in self.experiments:
            # Create the modified config for this experiment
            config = e.func(self.base_config)
            config.outputs.html.title = e.title

            # Update the paths for the output files
            if config.outputs.html.filename:
                config.outputs.html.filename = self._rename_file(
                    e.tag, config.outputs.html.filename
                )
            if config.outputs.pickle:
                config.outputs.pickle = self._rename_file(e.tag, config.outputs.pickle)
            if config.outputs.excel:
                config.outputs.excel = self._rename_file(e.tag, config.outputs.excel)

            summary = None

            if e.external:
                # If an external file is provided, load it and skip the simulation.
                # This is done without regard for the use_existing parameter, and
                # the absence of the external file is always an error.
                summary = SimulationTables.from_pickle(e.external)
                print(f"Loaded {e.tag} from {e.external}")

            if use_existing:
                # Check if the output pickle files already exist
                try:
                    summary = SimulationTables.from_pickle(config.outputs.pickle)
                except FileNotFoundError:
                    if use_existing == "raise":
                        raise
                    elif use_existing == "ignore":
                        continue
                else:
                    # If we reach this point, we have successfully loaded the
                    # output pickle file
                    print(f"Loaded {e.tag} from {config.outputs.pickle}")

            if summary is None:
                # If we reach this point, we need to run the simulation

                # Initialize the simulation
                if e.multi:
                    sim = MultiSimulation(config)
                else:
                    sim = Simulation(config)

                # Run the simulation
                summary = sim.run()

                # Clean up the simulation
                del sim

            results[e.tag] = summary

        return results
