#
# Simulate marketing actions, such as fare sales
#  - Can start / stop at specified a sample
#  - Can be specific to a limited geo-region
#  - Can specify change of base_demand ad/or arrival curve (demand stimulation)
#

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

from ._common import RmAction

if TYPE_CHECKING:
    from passengersim.driver import Simulation


class MarketAction(RmAction):
    frequency: Literal["dcp", "daily", "daily_pre_dep", "non_dcp", "begin_sample", "end_sample", "weekly"] = (
        "end_sample"
    )

    start_sample: int
    end_sample: int
    carrier: str | list[str] | None = None
    orig_airport: str | list = []
    dest_airport: str | list = []
    orig_state: str | list | None = []
    dest_state: str | list | None = []
    fare_change_pct: float = 0.0
    fare_classes: str | list | None = None
    demand_change_pct: float = 0.0
    demand_segment: str | list | None = None
    demand_arrival_curve: str = ""
    places: Any = None

    def __init__(
        self,
        *,
        name: str = "",
        carrier: str = "",
        orig_airport: str | list = None,
        dest_airport: str | list = None,
        orig_state: str | list | None = None,
        dest_state: str | list | None = None,
        fare_change_pct: float = 0.0,
        fare_classes: str | list | None = None,
        demand_change_pct: float = 0.0,
        demand_segment: str | list | None = None,
        demand_arrival_curve: str = "",
        start_sample: int = 0,
        end_sample: int = 0,
    ):
        self.name = name
        self.carrier = carrier
        self.orig_airport = orig_airport
        self.dest_airport = dest_airport
        self.orig_state = orig_state
        self.dest_state = dest_state
        self.fare_change_pct = fare_change_pct
        self.fare_classes = fare_classes
        self.demand_change_pct = demand_change_pct
        self.demand_segment = demand_segment
        self.demand_arrival_curve = demand_arrival_curve
        self.start_sample = start_sample
        self.end_sample = end_sample

        # User can specify just one airport or state,
        # but I will make a list to simplify the code later on
        if isinstance(self.carrier, str) and len(self.carrier) > 0:
            self.carrier = [self.carrier]
        if isinstance(self.orig_airport, str) and len(self.orig_airport) > 0:
            self.orig_airport = [self.orig_airport]
        if isinstance(self.dest_airport, str) and len(self.dest_airport) > 0:
            self.dest_airport = [self.dest_airport]
        if isinstance(self.orig_state, str) and len(self.orig_state) > 0:
            self.orig_state = [self.orig_state]
        if isinstance(self.dest_state, str) and len(self.dest_state) > 0:
            self.dest_state = [self.dest_state]
        if isinstance(self.fare_classes, str) and len(self.fare_classes) > 0:
            self.fare_classes = [self.fare_classes]
        if isinstance(self.demand_segment, str) and len(self.demand_segment) > 0:
            self.demand_segment = [self.demand_segment]

    def use_config(self, cfg):
        """We use the places data to lookup state codes, countries, etc."""
        self.places = cfg.places
        if self.orig_state is not None and len(self.orig_state) and self.places is None:
            raise KeyError("MarketAction has state codes, but the config has no places")
        if self.dest_state is not None and len(self.dest_state) and self.places is None:
            raise KeyError("MarketAction has state codes, but the config has no places")

    def run(self, sim: Simulation, days_prior: int):
        """Simulate marketing department changes"""
        sample = sim.sample
        if sample != self.start_sample and sample != self.end_sample:
            return 0

        num_changes = 0
        if self.demand_change_pct != 0.0:
            for dmd in sim.demands:
                num_changes += self.adjust_demand(sample, dmd)

        if self.fare_change_pct != 0.0:
            for fare in sim.fares:
                num_changes += self.adjust_fare(sample, fare)

        return num_changes  # We mostly use this in unit testing

    def adjust_demand(self, sample, dmd):
        if self.orig_airport is not None and len(self.orig_airport) > 0 and dmd.orig not in self.orig_airport:
            return 0
        if self.dest_airport is not None and len(self.dest_airport) > 0 and dmd.dest not in self.dest_airport:
            return 0

        if self.orig_state is not None and len(self.orig_state) > 0:
            if dmd.orig not in self.places:
                raise KeyError(f"MarketAction has state codes, but Demand orig: {dmd.orig} not found in places")
            dmd_orig_state = self.places[dmd.orig].state
            if dmd_orig_state not in self.orig_state:
                return 0

        if self.dest_state is not None and len(self.dest_state) > 0:
            if dmd.dest not in self.places:
                raise KeyError(f"MarketAction has state codes, but Demand dest: {dmd.dest} not found in places")
            dmd_dest_state = self.places[dmd.dest].state
            if dmd_dest_state not in self.dest_state:
                return 0

        if self.demand_segment is not None and self.demand_segment != 0 and dmd.segment not in self.demand_segment:
            return 0

        if abs(self.demand_change_pct) < 1.0 or abs(self.demand_change_pct) > 100.0:
            warnings.warn("demand_change_pct is a percentage, use 10.0 of you want 10%", stacklevel=2)
        tmp = self.demand_change_pct * 0.01
        if sample == self.start_sample:
            dmd.base_demand *= 1.0 + tmp
        else:
            dmd.base_demand /= 1.0 + tmp

        return 1

    def adjust_fare(self, sample, fare):
        if self.carrier is not None and len(self.carrier) > 0 and fare.carrier not in self.carrier:
            return 0
        if self.orig_airport is not None and len(self.orig_airport) > 0 and fare.orig not in self.orig_airport:
            return 0
        if self.dest_airport is not None and len(self.dest_airport) > 0 and fare.dest not in self.dest_airport:
            return 0
        if self.fare_classes is not None and len(self.fare_classes) > 0 and fare.booking_class not in self.fare_classes:
            return 0

        if self.orig_state is not None and len(self.orig_state) > 0:
            if fare.orig not in self.places:
                raise KeyError(f"MarketAction has state codes, but Fare orig: {fare.orig} not found in places")
            fare_orig_state = self.places[fare.orig].state
            if fare_orig_state not in self.orig_state:
                return 0

        if self.dest_state is not None and len(self.dest_state) > 0:
            if fare.dest not in self.places:
                raise KeyError(f"MarketAction has state codes, but Fare dest: {fare.dest} not found in places")
            fare_dest_state = self.places[fare.dest].state
            if fare_dest_state not in self.dest_state:
                return 0

        if abs(self.fare_change_pct) < 1.0 or abs(self.fare_change_pct) > 100.0:
            warnings.warn("fare_change_pct is a percetange, use 10.0 of you want 10%", stacklevel=2)
        tmp = self.fare_change_pct * 0.01
        if sample == self.start_sample:
            prev = fare.price
            fare.price *= 1.0 + tmp
            print(f"Adjusting fare: {fare.orig}-{fare.dest}:{fare.booking_class}, ${prev} -> ${fare.price}")
        else:
            fare.price /= 1.0 + tmp

        return 1
