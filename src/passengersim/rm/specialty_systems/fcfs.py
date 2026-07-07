from __future__ import annotations

from typing import TYPE_CHECKING

from passengersim.rm.standard_forecasting import StandardLegForecast
from passengersim.rm.systems import RmAction, RmSys, register_rm_system
from passengersim.rm.untruncation import LegDetruncation

if TYPE_CHECKING:
    from passengersim.driver import Simulation


class NoProtection(RmAction):
    """A no-protection RM action.

    This RM action does not do anything, and is used to indicate that the lack of
    any values set as bucket protection levels is intentional. Otherwise, PassengerSim
    will flag this as an error.
    """

    requires: set[str] = {}
    produces: set[str] = {"bucket_allocations"}
    frequency = "begin_sample"

    def run(self, sim: Simulation, days_prior: int):
        pass


@register_rm_system
class FirstComeFirstServed(RmSys):
    """A first come, first served RM system.

    First come first served (FCFS) is a simple method for allocating capacity to
    customers, and it operates pretty much as you would expect: customers whom
    arrive first are offered products, no attempt is made to optimize for anything.

    This process of capacity allocation will also occur if no RM optimization
    algorithm is applied, but the explicit system allow the user to be intentional
    about selecting this algorithm. This intentionality is enforced by PassengerSim,
    as actually having no RM system is an error. This RM system allows the user to
    explicitly select a no-optimization RM system.

    This RM system does implement leg-level untruncation and forecasting, but does
    not *do* any with the resulting forecasts.
    """

    _name = "FCFS"
    """This RM system is registered under the name "FCFS"."""

    availability_control = "leg"
    """This RM system uses leg-level class allocation availability controls."""

    actions = [
        LegDetruncation,
        StandardLegForecast.configure(
            fixed=dict(
                algorithm="additive_pickup",
            ),
        ),
        NoProtection,
    ]
