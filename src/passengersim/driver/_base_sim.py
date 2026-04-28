#
# Driver program to load a simulation from YAML, run it and return results
# (c) PassengerSim LLC
#

from __future__ import annotations

import logging
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

import passengersim.config.rm_systems
import passengersim.core
from passengersim.config import Config
from passengersim.core import (
    Market,
    SimulationEngine,
)
from passengersim.utils.filenaming import timestamp_now
from passengersim.utils.tempdir import MaybeTemporaryDirectory

if TYPE_CHECKING:
    pass

logger = logging.getLogger("passengersim")


class BaseSimulation(ABC):
    @classmethod
    def from_yaml(
        cls,
        filenames: pathlib.Path | list[pathlib.Path],
        output_dir: pathlib.Path | None = None,
    ):
        """
        Create a Simulation object from a YAML file.

        Parameters
        ----------
        filenames : pathlib.Path | list[pathlib.Path]
        output_dir : pathlib.Path | None, optional

        Returns
        -------
        Simulation
        """
        config = passengersim.config.Config.from_yaml(filenames)
        return cls(config, output_dir)

    def __init__(
        self,
        config: Config,
        output_dir: pathlib.Path | None = None,
    ):
        """
        Initialize a BaseSimulation instance.

        Parameters
        ----------
        config : Config
            The simulation configuration object.
        output_dir : pathlib.Path or None, optional
            Directory for output files. If None, a temporary directory
            will be created automatically.
        """
        self.cnx = None
        self.output_dir = MaybeTemporaryDirectory(output_dir)

        # establish a common "timestamp" for this Simulation, which is the date
        # and time the Simulation object is created.  This timestamp will be
        # used to potentially tag multiple output files, so having a single
        # common timestamp set here allows us to synchronize output names.
        self._timestamp = timestamp_now()

    @property
    @abstractmethod
    def _eng(self) -> SimulationEngine:
        """
        Access to the underlying simulation engine.

        Returns
        -------
        SimulationEngine
            The core simulation engine instance.

        Notes
        -----
        This is an abstract property that must be implemented by subclasses.
        """
        raise NotImplementedError

    def path_names(self):
        """
        Get a mapping of path IDs to path names.

        Returns
        -------
        dict
            Dictionary mapping path IDs to string representations of paths.
        """
        result = {}
        for p in self._eng.paths:
            result[p.path_id] = str(p)
        return result

    @property
    def markets(self) -> Mapping[str, Market]:
        """
        Access markets in the simulation.

        Returns
        -------
        Mapping[str, Market]
            A mapping of market names to Market objects.
        """
        return self._eng.markets

    @property
    def paths(self):
        """
        Generator of all paths in the simulation.

        Returns
        -------
        generator
            Generator yielding path objects from the simulation.
        """
        return self._eng.paths

    @property
    def pathclasses(self):
        """
        Generator of all path classes in the simulation.

        Yields
        ------
        pathclass
            Path class objects from all paths in the simulation.
        """
        for path in self._eng.paths:
            yield from path.pathclasses

    def pathclasses_for_carrier(self, carrier: str):
        """
        Generator of all path classes for a given carrier.

        Parameters
        ----------
        carrier : str
            The carrier name to filter path classes by.

        Yields
        ------
        pathclass
            Path class objects for the specified carrier.
        """
        for path in self._eng.paths:
            if path.carrier_name == carrier:
                yield from path.pathclasses

    @property
    def demands(self):
        """
        Generator of all demands in the simulation.

        Returns
        -------
        DemandIterator
            Iterator object for accessing demand data.
        """
        from passengersim.iterators.demand import DemandIterator

        return DemandIterator(self._eng)

    @property
    def fares(self):
        """
        Generator of all fares in the simulation.

        Returns
        -------
        FareIterator
            Iterator object for accessing fare data.
        """
        from passengersim.iterators.fare import FareIterator

        return FareIterator(self._eng)
