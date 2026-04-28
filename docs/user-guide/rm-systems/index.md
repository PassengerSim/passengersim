# RM Systems

A revenue management (RM) system, defined in python as an
[`RmSys`](passengersim.rm.systems.RmSys), has two main components:
actions and availability controls.

The **actions** are a sequence of steps, each defined as an
[`RmAction`][passengersim.rm.systems.RmAction] that are executed periodically, as
often as daily, although actions can be set to run only on DCPs or at other less than
daily frequencies. These actions can include demand untruncation, demand forecasting,
optimization, or other arbitrary manipulations.  Actions are highly customizable,
and can be written by users to implement their own algorithms if desired.

The **availability controls** define how product availability is managed during
the core simulation.  That is, as each potential customer arrives, the availability
controls determine which products are made available to that customer based on the
current state of bookings, fare class protection levels, bid prices, etc.
Availability controls have only limited customization options, as they are coded in
the high-performance core simulation loop.

Each carrier should have an RM system that it uses. In PassengerSim, users have
the ability to create a single RM system framework and assign it to all carriers, or to
create multiple RM systems and assign different RM systems to different carriers.

## Standard RM Systems

PassengerSim includes several standard RM systems that can be used out-of-the-box.  These
standard RM systems are designed to reflect common RM systems deployed in "real world"
applications, and include commonly used algorithms for untruncation, forecasting, and
optimization.  The standard RM systems are defined in the
[`passengersim.rm.standard_systems`][passengersim.rm.standard_systems] module, and include:

- [`E`][passengersim.rm.standard_systems.E]: A basic RM system
  that uses leg-level untruncation, standard leg forecasting, and EMSR-B optimization
  with leg-level availability controls.
- [`P`][passengersim.rm.standard_systems.P]: A path-based RM system
  that uses path-level untruncation, standard path forecasting, and probabilistic bid
  price (ProBP) optimization with bid price availability controls.
- [`U`][passengersim.rm.standard_systems.U]: A path-based RM system
  that uses path-level untruncation, standard path forecasting, and unbucketed
  dynamic programming optimization with bid price availability controls.
- [`M`][passengersim.rm.standard_systems.M]: A path-based RM system for partly or
  fully unconstrained networks, which employs path-level untruncation of yieldable
  demand, hybrid conditional path forecasting, and probabilistic bid
  price (ProBP) optimization with bid price availability controls.
- [`V`][passengersim.rm.standard_systems.M]: Similar to `M`, but using unbucketed
  dynamic programming optimization.

Each of these standard RM systems can be assigned to carriers within the simulation
configuration simply by specifying the RM system's name in the carrier configuration.
For example, the configuration input shown below could be used to assign the `E` RM
system to one carrier, and the `U` RM system to another carrier.

```yaml title="simulation_config.yaml"
carriers:
  AL1:
    rm_system: E
  AL2:
    rm_system: U
```

Each standard RM system can also be customized by specifying options for the
individual actions within the RM system.  See the documentation for each standard
RM system for details on the available options.



## Defining New RM Systems

RM systems are defined in Python by creating a new class that inherits from
[`RmSys`][passengersim.rm.systems.RmSys] and decorating it with the
[`register_rm_system`][passengersim.rm.systems.register_rm_system] decorator.  The
class must define:

- an [`actions`][passengersim.rm.systems.RmSys.actions] attribute, which is a list of
the actions to be performed in sequence.  Each action is defined by a class that
inherits from [`RmAction`][passengersim.rm.systems.RmAction].  Many standard actions
are provided in PassengerSim, including untruncation, forecasting, and optimization
actions.

- an [`availability_control`][passengersim.rm.systems.RmSys.availability_control]
attribute, which is a string defining the type of
availability control to be used.  The most commonly used options are `"leg"` for
leg-level bucket protection controls by fare class, and `"bp"` for bid price
controls. (Other options are also available, see the documentation for details.)

```python title="basic_rm_system.py"
from passengersim.rm.systems import RmSys, RmSysOption, register_rm_system
from passengersim.rm.untruncation import LegUntruncation
from passengersim.rm.standard_forecasting import StandardLegForecast
from passengersim.rm.emsr import ExpectedMarginalSeatRevenue

@register_rm_system #(3)!
class BasicRmSys(RmSys):
    actions = [ #(1)!
        LegUntruncation,
        StandardLegForecast,
        ExpectedMarginalSeatRevenue
    ]
    availability_control = "leg" #(2)!
```
{ .annotate }

1.  This basic RM system includes three actions:
    [untruncation][passengersim.rm.untruncation.LegUntruncation],
    [forecasting][passengersim.rm.standard_forecasting.StandardLegForecast], and
    [EMSR optimization][passengersim.rm.emsr.ExpectedMarginalSeatRevenue].  There
    are no custom options specified here, so the default options for each action
    will be used.  Those defaults can be ascertained by examining the documentation
    for each action.
2.  Setting the `availability_control` to `leg` tells PassengerSim that a
    carrier using this `RmSys` will set fare class allocation limits at the leg
    bucket level.
3.  The [`register_rm_system`][passengersim.rm.systems.register_rm_system] decorator
    makes this RM system available for use within PassengerSim.  Without this decorator,
    the RM system will not be discoverable by PassengerSim, and cannot be used.


## Configurable Options

The example above defines a basic RM system with default options for each action.
As defined, it cannot be configured at all for individual carriers; and carrier
using this system will always use the same default options.

RM systems can also be defined to allow configuration of options for individual
carriers, or across different simulations.  This is done by using the
[`configure`][passengersim.rm.systems.RmAction.configure]
method on each action within the RM system's `actions` list, and by using the
[`RmSysOption`](passengersim.rm._system_options.RmSysOption) class to define options
that can be set when the RM system is assigned to a carrier.

```python title="basic_rm_system.py"
from passengersim.rm.systems import RmSys, RmSysOption, register_rm_system
from passengersim.rm.untruncation import LegUntruncation
from passengersim.rm.standard_forecasting import StandardLegForecast
from passengersim.rm.emsr import ExpectedMarginalSeatRevenue

@register_rm_system
class BasicRmSys2(RmSys):
    actions = [
        LegUntruncation.configure( #(1)!
            algorithm=RmSysOption("em_algorithm", default="em"),
            maxiter=RmSysOption("em_maxiter", expected_type=int, default=10),
            fixed=dict(initialization_method="pods"), #(2)!
        ),
        StandardLegForecast,
        ExpectedMarginalSeatRevenue
    ]
    availability_control = "leg"
```
{ .annotate }

1.  The `configure` method is used here to specify that the
    `LegUntruncation` action should have some configurable options. Each
    keyword argument to `configure` corresponds to an attribute on the action
    class that can be set.  The value of each argument is an instance of
    `RmSysOption`, which defines the name of the option as it will appear in the
    RM system configuration, and optionally the expected type of the option value,
    and a default value to be used if the option is not specified.
2.  The `fixed` argument on the `configure` method is special-cased, and is not
    an `RmSysOption`.  Instead, it is simply a dictionary of attribute names and
    values that will always be set on the action when it is used.  This allows
    certain attributes of the action to be set to fixed values that cannot be
    changed by the user, but also may be different from the default values.


## Using RM Systems

Once an RM system has been defined and registered, it can be assigned to carriers
within the simulation configuration.  This is done by specifying the RM system's
registered name (typically the class name) in the carrier configuration. For example,
the configuration input shown below could be used to assign the `BasicRmSys` RM system
defined earlier to two different carriers.

```yaml title="simulation_config.yaml"
carriers:
  AL1:
    rm_system: BasicRmSys
  AL2:
    rm_system: BasicRmSys #(1)!
```
{ .annotate }

1.  Both carriers are assigned to use the same `BasicRmSys` RM system
    defined earlier. They each will have a separate _instance_ of the RM system, but
    both instances will use the same setup defined in the `BasicRmSys` class.

To use an RM system with configurable options, the configuration could look like this,
which assigns the `BasicRmSys2` RM system defined earlier to two different carriers.

```yaml title="simulation_config.yaml"
carriers:
  AL1:
    rm_system: BasicRmSys2
    rm_system_options: #(1)!
      em_algorithm: em
      em_maxiter: 20
  AL2:
    rm_system: BasicRmSys2 #(2)!
```
{ .annotate }

1.  Carrier `AL1` has specified custom options for the untruncation action, setting
    the `em_algorithm` to `"em"` and the `em_maxiter` to `20`.
2.  Carrier `AL2` has no options set, so it will use the default values for all
    options defined in the RM system.
