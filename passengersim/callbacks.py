from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import TYPE_CHECKING

from passengersim_core import (
    Event,
)

from .tracers.generic import GenericTracer

if TYPE_CHECKING:
    from passengersim_core import SimulationEngine


class CallbackMixin:
    if TYPE_CHECKING:
        sim: SimulationEngine

    def end_sample_callback(self, callback: Callable[[CallbackMixin], None] | GenericTracer):
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
        if isinstance(callback, GenericTracer):
            callback = callback.fresh()
        self.end_sample_callbacks.append(callback)
        return callback

    def begin_sample_callback(self, callback: Callable[[CallbackMixin], None] | GenericTracer):
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
        if isinstance(callback, GenericTracer):
            callback = callback.fresh()
        self.begin_sample_callbacks.append(callback)
        return callback

    def daily_callback(self, callback: Callable[[CallbackMixin, int], None] | GenericTracer):
        """Register a function to be triggered each day during a sample.

        The callback function will be triggered after all RM steps when the day
        coincides with a DCP.

        Parameters
        ----------
        callback : Callable[[Simulation, int], None]
            The callback function to register.  It should accept two arguments,
            which will be the Simulation object and the days_prior, and return None.
        """
        if not hasattr(self, "daily_callbacks"):
            self.daily_callbacks = []
        if isinstance(callback, GenericTracer):
            callback = callback.fresh()
        self.daily_callbacks.append(callback)
        return callback

    def callback_functions(self) -> dict[str, list[Callable]]:
        """Get all callback functions."""
        cb = {}
        for k in ["begin_sample", "end_sample", "daily"]:
            if hasattr(self, f"{k}_callbacks"):
                cb[f"{k}_callbacks"] = getattr(self, f"{k}_callbacks", [])
        return cb

    def apply_callback_functions(self, sim: CallbackMixin):
        callbacks = self.callback_functions()
        for cb in callbacks.get("begin_sample_callbacks", []):
            sim.begin_sample_callback(cb)
        for cb in callbacks.get("end_sample_callbacks", []):
            sim.end_sample_callback(cb)
        for cb in callbacks.get("daily_callbacks", []):
            sim.daily_callback(cb)

    def add_callback_events(self):
        """Add callback events to the simulation event queue."""
        dcp_hour = self.sim.config.simulation_controls.dcp_hour

        for callback in getattr(self, "begin_sample_callbacks", []):
            dcp = self.dcp_list[0]
            # no customers can arrive within 5 seconds of a DCP.
            # we want these callbacks to be triggered after the first DCP
            # but before any customers can arrive, so we add one second.
            event_time = int(self.sim.base_time - dcp * 86400 + 3600 * dcp_hour) + 1
            rm_event = Event(
                (
                    "callback_begin_sample",
                    callback,
                ),
                event_time,
            )
            self.sim.add_event(rm_event)

        for callback in getattr(self, "end_sample_callbacks", []):
            # we want these callbacks to be triggered after the last DCP
            # so we add one second.
            event_time = int(self.sim.base_time + 3600 * dcp_hour) + 1
            rm_event = Event(
                (
                    "callback_end_sample",
                    callback,
                ),
                event_time,
            )
            self.sim.add_event(rm_event)

        for callback in getattr(self, "daily_callbacks", []):
            day = self.dcp_list[0]
            # The priority is a number of seconds before or after the overnight
            # trigger time.  If priority is negative, the daily callback will be
            # run before any RM system daily or same-time DCP events. If priority
            # is positive it will be run after other RM events.
            priority = getattr(callback, "priority", 0)
            while day >= 0:
                event_time = int(self.sim.base_time - day * 86400 + 3600 * dcp_hour + priority)
                rm_event = Event(("callback_daily", callback, day), event_time)
                self.sim.add_event(rm_event)
                day -= 1


def _flatten_dict_keys(d):
    for k, v in d.items():
        if isinstance(v, dict):
            for sub_k, sub_v in _flatten_dict_keys(v):
                yield f"{k}.{sub_k}", sub_v
        elif isinstance(v, list | tuple):
            for i, sub_v in enumerate(v):
                yield f"{k}[{i}]", sub_v
        else:
            yield k, v


def _flatten_iter_of_dict(i):
    for item in i:
        yield dict(_flatten_dict_keys(item))


class CallbackData(MutableMapping):
    """Data collected during callbacks."""

    def __init__(self):
        self._data = {}

    def get_data(self, label: str, trial: int, sample: int, days_prior: int | None = None):
        key_match = {"trial": trial, "sample": sample}
        if days_prior is not None:
            key_match["days_prior"] = days_prior
        if label not in self._data:
            self._data[label] = [key_match]
        store = self._data[label][-1]
        if any(store.get(k) != v for k, v in key_match.items()):
            self._data[label].append(key_match)
            store = self._data[label][-1]
        return store

    def update_data(
        self,
        label: str,
        trial: int,
        sample: int,
        days_prior: int | None = None,
        **kwargs,
    ):
        store = self.get_data(label, trial, sample, days_prior)
        store.update(kwargs)

    def to_dataframe(self, item: str):
        import pandas as pd

        if item in self._data:
            return pd.DataFrame(_flatten_iter_of_dict(self._data[item]))
        else:
            raise KeyError(f"{item} not found in callback data")

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, item):
        if not item.startswith("_") and item in self._data:
            return self._data[item]
        raise AttributeError(f"{self.__class__.__name__}" f" has no attribute '{item}'")

    def __repr__(self):
        if self._data:
            keys = ", ".join(self._data.keys())
            return f"<{self.__class__.__module__}.{self.__class__.__name__} from {keys}>"
        else:
            return f"<{self.__class__.__module__}.{self.__class__.__name__} with no data>"

    def __bool__(self):
        return bool(self._data)

    def __add__(self, other):
        if isinstance(other, CallbackData):
            new = CallbackData()
            for k in self._data:
                new._data[k] = self._data[k]
                if k in other._data:
                    new._data[k] += other._data[k]
            for k in other._data:
                if k not in self._data:
                    new._data[k] = other._data[k]
            return new
        elif isinstance(other, int) and other == 0:
            return self
        elif other is None:
            return self
        else:
            return NotImplemented

    def __radd__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        elif other is None:
            return self
        else:
            return NotImplemented

    def __dir__(self):
        return self._data.keys()
