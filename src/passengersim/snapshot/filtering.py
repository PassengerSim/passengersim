from __future__ import annotations

import pathlib
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, field_validator

from passengersim.snapshot.instruction import SnapshotInstruction

if TYPE_CHECKING:
    from passengersim.core import Leg
    from passengersim.driver import Simulation


class GenericSnapshotFilter(ABC, BaseModel, validate_assignment=True):
    @abstractmethod
    def resolve(self, sim: Simulation, days_prior: int, *args, **kwargs) -> SnapshotInstruction | None:
        raise NotImplementedError


class NetworkSnapshotFilter(GenericSnapshotFilter):
    """A generic snapshot filter for grabbing snapshots on the whole network at specific moments."""

    title: str = ""
    """The title of the snapshot, will be printed near the top of each snapshot."""

    trial: set[int] = {0}
    """Only trigger this snapshot on these trials."""

    sample: set[int] = {99}
    """Only trigger this snapshot on these samples."""

    days_prior: set[int] = {63}
    """Only trigger this snapshot on these days prior."""

    directory: pathlib.Path | None = None
    """The directory where snapshots will be saved.

    If None, the snapshot content will be printed to stdout."""

    @field_validator("trial", "sample", "days_prior", mode="before")
    def _allow_singletons(cls, v):
        """Allow a singleton value that is converted to a set of one item."""
        if not isinstance(v, list | tuple | set):
            v = {v}
        return v

    def _get_filepath(self, sim: Simulation, days_prior: int) -> pathlib.Path | None:
        if self.directory is None:
            return None
        pth = self.directory
        if len(self.days_prior) > 1:
            pth = pth.joinpath(f"days_prior-{days_prior}")
        if len(self.trial) > 1 and sim.num_trials > 1:
            pth = pth.joinpath(f"trial-{sim.trial}")
        if len(self.sample) > 1:
            pth = pth.joinpath(f"sample-{sim.sample}")
        if pth == self.directory:
            pth = pth.joinpath("snapshot")
        pth.parent.mkdir(parents=True, exist_ok=True)
        return pth.with_suffix(".log")

    def resolve(self, sim: Simulation, days_prior: int) -> SnapshotInstruction | None:
        """Resolve whether to generate a snapshot or not.

        Parameters
        ----------
        sim : Simulation
            The simulation being run.  Used to filter on the current trial and sample.
        days_prior : int
            The number of days prior to departure for the current snapshot moment.
            Used to filter on the current days prior.
        """
        # Check the filter conditions
        info = f"{self.title}\n"

        # If not the right trial, sample, dcp, or leg_id, then don't trigger

        if len(self.trial) > 0 and sim.trial not in self.trial and sim.num_trials > 1:
            return None

        if len(self.sample) > 0 and sim.sample not in self.sample:
            return None

        if len(self.days_prior) > 0 and days_prior not in self.days_prior:
            return None

        info += f"  trial      = {sim.trial}"
        info += f"\n  sample     = {sim.sample}"
        info += f"\n  days_prior = {sim.last_dcp}"

        # Now do something
        snapshot_file = self._get_filepath(sim, days_prior)
        created_date = time.strftime("Snapshot created %Y-%m-%d %A %I:%M:%S %p")
        header = f"{created_date}\n{info}\n\n"
        if len(self.title) > 0 and not snapshot_file:
            print(info, flush=True)

        return SnapshotInstruction(True, snapshot_file, why=header)


class LegSnapshotFilter(NetworkSnapshotFilter):
    """A generic snapshot filter for grabbing snapshots on legs at specific moments."""

    leg_ids: set[int] = set()
    """Only trigger this snapshot on these legs.

    If you want to filter on carrier, origin, destination, flt_no, or any other leg attribute,
    just figure out which leg_id's those are before the simulation, and filter on them.
    """

    def _get_filepath(self, sim: Simulation, days_prior: int, leg: Leg) -> pathlib.Path | None:
        if self.directory is None:
            return None
        pth = self.directory
        if len(self.leg_ids) > 1:
            pth = pth.joinpath(f"leg_id-{leg.leg_id}")
        if len(self.days_prior) > 1:
            pth = pth.joinpath(f"days_prior-{days_prior}")
        if len(self.trial) > 1 and sim.num_trials > 1:
            pth = pth.joinpath(f"trial-{sim.trial}")
        if len(self.sample) > 1:
            pth = pth.joinpath(f"sample-{sim.sample}")
        if pth == self.directory:
            pth = pth.joinpath("snapshot")
        pth.parent.mkdir(parents=True, exist_ok=True)
        return pth.with_suffix(".log")

    def resolve(self, sim: Simulation, days_prior: int, leg: Leg) -> SnapshotInstruction | None:
        """Resolve whether to generate a snapshot or not.

        Parameters
        ----------
        sim : Simulation
            The simulation being run.  Used to filter on the current trial and sample.
        days_prior : int
            The number of days prior to departure for the current snapshot moment.
            Used to filter on the current days prior.
        leg : Leg
            The leg being processed.  Used to filter on the leg_id, and also to provide
            information for the snapshot content if triggered.
        """
        # First, if there are no leg_ids specified, then this filter doesn't apply to any legs,
        # so don't trigger and don't even return an instruction explaining why, since this is
        # effectively a no-op filter
        if len(self.leg_ids) == 0:
            return None

        # Check the filter conditions
        info = f"{self.title}\n"

        # If not the right trial, sample, dcp, or leg_id, then don't trigger

        if len(self.trial) > 0 and sim.trial not in self.trial and sim.num_trials > 1:
            return None

        if len(self.sample) > 0 and sim.sample not in self.sample:
            return None

        if len(self.days_prior) > 0 and days_prior not in self.days_prior:
            return None

        if leg.leg_id not in self.leg_ids:
            return None

        info += f"  trial      = {sim.trial}"
        info += f"\n  sample     = {sim.sample}"
        info += f"\n  days_prior = {sim.last_dcp}"
        info += f"\n  leg_id     = {leg.leg_id}"
        info += f"\n  carrier    = {leg.carrier}"
        info += f"\n  leg orig   = {leg.orig}"
        info += f"\n  leg dest   = {leg.dest}"
        info += f"\n  flt_no     = {leg.flt_no}"

        # Now do something
        snapshot_file = self._get_filepath(sim, days_prior, leg)
        created_date = time.strftime("Snapshot created %Y-%m-%d %A %I:%M:%S %p")
        header = f"{created_date}\n{info}\n\n"
        if len(self.title) > 0 and not snapshot_file:
            print(info, flush=True)

        return SnapshotInstruction(True, snapshot_file, why=header)
