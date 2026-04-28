from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from passengersim import SimulationEngine


class DemandIterator:
    """An iterator for demands that allows filtering on attributes.

    Parameters
    ----------
    eng : SimulationEngine
        The simulation engine.
    orig : str, optional
        The origin to filter on.
    dest : str, optional
        The destination to filter on.
    segment : str, optional
        The segment to filter on.
    """

    def __init__(
        self,
        eng: SimulationEngine,
        orig: str | None = None,
        dest: str | None = None,
        segment: str | None = None,
    ):
        self._eng = eng
        self._demands_iter = iter(self._eng.demands)
        self._orig = orig
        self._dest = dest
        self._segment = segment

    def __iter__(self):
        self._demands_iter = iter(self._eng.demands)
        return self

    def __next__(self):
        while True:
            dmd = next(self._demands_iter)
            if self._orig is not None and dmd.orig != self._orig:
                continue
            if self._dest is not None and dmd.dest != self._dest:
                continue
            if self._segment is not None and dmd.segment != self._segment:
                continue
            return dmd

    def __call__(self, **kwargs):
        kw = dict(
            orig=self._orig,
            dest=self._dest,
            segment=self._segment,
        )
        kw.update(kwargs)
        return DemandIterator(self._eng, **kw)

    def select(self, **kwargs):
        i = self(**kwargs)
        return next(i)
