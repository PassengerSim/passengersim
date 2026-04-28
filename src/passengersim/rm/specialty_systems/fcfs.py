from passengersim.rm.standard_forecasting import StandardLegForecast
from passengersim.rm.systems import RmSys, register_rm_system
from passengersim.rm.untruncation import LegUntruncation


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
        LegUntruncation,
        StandardLegForecast.configure(
            fixed=dict(
                algorithm="additive_pickup",
            ),
        ),
    ]
