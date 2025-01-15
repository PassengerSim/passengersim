from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from passengersim_core import (
    Event,
)

if TYPE_CHECKING:
    from passengersim_core import SimulationEngine


class CallbackMixin:
    if TYPE_CHECKING:
        sim: SimulationEngine

    def end_sample_callback(self, callback: Callable[[CallbackMixin], None]):
        """Register a function to be triggered at the end of each sample.

        The callback function will be triggered before counters are reset or
        history buffers are rolled over.

        Parameters
        ----------
        callback : Callable[[Simulation], None]
            The callback function to register.  It should accept a single argument,
            which will be the Simulation object, and return None.
        """
        if not hasattr(self, "end_sample_callbacks"):
            self.end_sample_callbacks = []
        self.end_sample_callbacks.append(callback)
        return callback

    def begin_sample_callback(self, callback: Callable[[CallbackMixin], None]):
        """Register a function to be triggered at the beginning of each sample.

        The callback function will be triggered after initial setup including
        all RM steps for the initial DCP, but before any customers can arrive.

        Parameters
        ----------
        callback : Callable[[Simulation], None]
            The callback function to register.  It should accept a single argument,
            which will be the Simulation object, and return None.
        """
        if not hasattr(self, "begin_sample_callbacks"):
            self.begin_sample_callbacks = []
        self.begin_sample_callbacks.append(callback)
        return callback

    def daily_callback(self, callback: Callable[[CallbackMixin, int], None]):
        """Register a function to be triggered each day during a sample.

        The callback function will be triggered after all RM steps when the day
        coincides with a DCP.

        Parameters
        ----------
        callback : Callable[[Simulation, int], None]
            The callback function to register.  It should accept a two argument,
            which will be the Simulation object and the days_prior, and return None.
        """
        if not hasattr(self, "daily_callbacks"):
            self.daily_callbacks = []
        self.daily_callbacks.append(callback)
        return callback

    def add_callback_events(self):
        """Add callback events to the simulation event queue."""
        dcp_hour = self.sim.config.simulation_controls.dcp_hour

        for callback in getattr(self, "begin_sample_callbacks", []):
            dcp = self.dcp_list[0]
            # no customers can arrive within 5 seconds of a DCP.
            # we want these callbacks to be triggered after the first DCP
            # but before any customers can arrive, so we add one second.
            event_time = int(self.sim.base_time - dcp * 86400 + 3600 * dcp_hour) + 1
            rm_event = Event((callback,), event_time)
            self.sim.add_event(rm_event)

        for callback in getattr(self, "end_sample_callbacks", []):
            # we want these callbacks to be triggered after the last DCP
            # so we add one second.
            event_time = int(self.sim.base_time + 3600 * dcp_hour) + 1
            rm_event = Event((callback,), event_time)
            self.sim.add_event(rm_event)

        for callback in getattr(self, "daily_callbacks", []):
            day = self.dcp_list[0]
            while day >= 0:
                event_time = int(self.sim.base_time - day * 86400 + 3600 * dcp_hour)
                rm_event = Event((callback, day), event_time)
                self.sim.add_event(rm_event)
                day -= 1
