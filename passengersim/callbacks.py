from __future__ import annotations

from collections.abc import Callable


class CallbackMixin:
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

    def trigger_end_sample_callbacks(self):
        """Trigger all registered end_sample_callbacks."""
        for callback in getattr(self, "end_sample_callbacks", []):
            callback(self)
