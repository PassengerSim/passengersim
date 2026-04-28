from __future__ import annotations

from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from passengersim.core import SimulationEngine


class Firehose:
    """Mixin for attaching and retrieving simulation firehose outputs.

    This helper expects subclasses to provide an ``eng`` attribute with the
    core firehose logger methods. Writable destinations are cached locally so
    buffered output can be retrieved after the engine flushes pending messages.
    """

    eng: SimulationEngine

    def __init__(self):
        self._firehose_buffers = {}
        """A mapping of firehose topic to buffer, for topics with a writable destination."""

    def attach_firehose(self, topic: str, destination: str | bool | TextIO | None) -> None:
        """Attach a firehose logger destination for a topic.

        Parameters
        ----------
        topic : str
            Firehose topic name to route from the simulation engine.
        destination : str or bool or file-like or None
            Destination passed through to the engine. Writable destinations are
            also stored locally so they can be returned by :meth:`get_firehose`.
        """
        if not isinstance(destination, str | bool | type(None)):
            # when adding a file-like object, attach here so we can get it back later if needed
            self._firehose_buffers[topic] = destination
        self.eng.attach_firehose_logger(topic, destination)

    def get_firehose(self, topic: str) -> TextIO | None:
        """Return the cached writable firehose destination for a topic.

        Parameters
        ----------
        topic : str
            Firehose topic name to look up.

        Returns
        -------
        TextIO or None
            The cached writable destination for ``topic``, if one was attached.

        Notes
        -----
        Pending firehose log output is flushed from the engine before the
        cached destination is returned.
        """
        self.eng.flush_firehoses()
        return self._firehose_buffers.get(topic)
