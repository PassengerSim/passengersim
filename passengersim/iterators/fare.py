from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from collections.abc import Collection

    from passengersim import SimulationEngine
    from passengersim.core import Carrier


class FareIterator:
    """An iterator for fares that allows filtering on attributes.

    Parameters
    ----------
    obj : hasattr(fares)
        The simulation engine or other object that has "fares", from which the
        fares shall be iterated.
    orig : str, optional
        The origin to filter on.
    dest : str, optional
        The destination to filter on.
    carrier : str or  Carrier, optional
        The carrier to filter on.
    days_prior : int, optional
        Only include fares where the advance purchase restriction (if any) is not
        violated at this number of days prior to departure.
    exclude_booking_classes : Collection[str], optional
        Exclude fares with these booking classes.
    """

    def __init__(
        self,
        sim: SimulationEngine,
        orig: str | None = None,
        dest: str | None = None,
        carrier: str | Carrier | None = None,
        days_prior: int | None = None,
        exclude_booking_classes: Collection[str] | None = None,
    ):
        self._obj = sim
        self._fare_iter = iter(self._obj.fares)
        self._orig = orig
        self._dest = dest
        if isinstance(carrier, str | None):
            self._carrier_name = carrier
        else:
            self._carrier_name = carrier.name
        self._days_prior = days_prior
        if isinstance(exclude_booking_classes, str):
            exclude_booking_classes = {exclude_booking_classes}
        if exclude_booking_classes is None:
            self._exclude_booking_classes = None
        else:
            self._exclude_booking_classes = set(exclude_booking_classes)

    def __iter__(self):
        self._fare_iter = iter(self._obj.fares)
        return self

    def __next__(self):
        while True:
            fare = next(self._fare_iter)
            if self._orig is not None and fare.orig != self._orig:
                continue
            if self._dest is not None and fare.dest != self._dest:
                continue
            if (
                self._carrier_name is not None
                and fare.carrier_name != self._carrier_name
            ):
                continue
            if self._days_prior is not None and self._days_prior < fare.adv_purch:
                continue
            if (
                self._exclude_booking_classes is not None
                and fare.booking_class in self._exclude_booking_classes
            ):
                continue
            return fare

    def __call__(self, **kwargs):
        kw = dict(
            orig=self._orig,
            dest=self._dest,
            carrier=self._carrier_name,
            days_prior=self._days_prior,
            exclude_booking_classes=self._exclude_booking_classes,
        )
        kw.update(kwargs)
        return FareIterator(self._obj, **kw)

    def select(self, **kwargs):
        i = self(**kwargs)
        return next(i)
