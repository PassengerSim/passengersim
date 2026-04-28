from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import dill

from ._common import RmAction, RmActionFactory
from ._system_options import RmSysOption  # noqa: F401

if TYPE_CHECKING:
    from passengersim.config import Config
    from passengersim.driver import Simulation


def _make_factory_queue(
    actions: Sequence[RmActionFactory | type[RmAction]], check_parameters: dict | None = None
) -> list[RmActionFactory]:
    """Get a list of RmActionFactory instances for this sequence of actions.

    For actions that are not already RmActionFactory instances, this method
    will call their `configure()` class method to create a factory that has
    no configurable parameters.
    """
    factory_queue = []
    for i in actions:
        if isinstance(i, RmActionFactory):
            factory_queue.append(i)
        elif isinstance(i, type) and issubclass(i, RmAction):
            factory_queue.append(i.configure())
        else:
            raise TypeError(
                f"Every item in RmSys.actions must be an RmAction class or a RmActionFactory instance; "
                f"found {type(i).__name__} instead."
            )
    return factory_queue


class RmSys:
    """A revenue management (RM) system that executes a sequence of RM actions."""

    priority: int = -1
    """Priority of this RM system when scheduled as an event callback.

    The default setting is -1, which means it will run just before other daily
    callbacks with the default priority of 0.
    """

    availability_control: Literal["leg", "cabin", "bp", "bp_loose", "classless", None] = None
    """The type of availability control used in this RM system.

    The selected availability control is injected into the Carrier object
    at the time it is created in the simulation, as this setting is used
    during the passenger arrival simulation loop, and instead of during the
    RM system steps run from the action queue each day.

    Subclasses of `RmSys` *must* set this class variable to one of the allowed
    values (other than None) to indicate the type of availability control
    used by that RM system.
    """

    actions: list[RmActionFactory | type[RmAction]] = []
    """List of RM actions (or action factories) to execute in this RM system.

    Each item in this list should be either an `RmActionFactory` instance,
    or an `RmAction` subclass.  If an `RmAction` subclass is provided,
    its `configure()` class method will be called to create an
    `RmActionFactory` with no configurable parameters.

    Subclasses of `RmSys` *must* set this class variable to a non-empty list
    of actions to be executed in order when this RM system runs.
    """

    def _check_action_parameters(self, factory_queue: Sequence[RmActionFactory], provided_parameters: dict) -> None:
        """Check that all provided parameters match at least one action factory."""
        for k in provided_parameters:
            if not any(f.has_option(k) for f in factory_queue):
                raise ValueError(
                    f"Keyword argument {k!r} does not match any configuration parameter in {self.__class__.__name__!r}."
                )

    def __init__(self, carrier: str, cfg: Config | None = None, **kwargs) -> None:
        """
        Initialize the RM system.

        Parameters
        ----------
        carrier : str
            The carrier code for which this RM system is configured.
        cfg : Config, optional
            The top level configuration object for the simulation, which is also
            used to help initialize this system. For example, the collection of
            data collection points (DCPs) given as days prior to departure, is used
            for steps that have a frequency of "DCP" to identify which days to run.
        **kwargs
            Additional keyword arguments to configure the RM actions in this system.

        Raises
        ------
        ValueError
            If `availability_control` is not defined, or if no actions are defined,
            or if any provided keyword argument does not match any action's configuration options.
        """
        if self.availability_control is None:
            raise ValueError(f"availability_control is not defined on {self.__class__.__name__}.")

        if not self.actions:
            raise ValueError(f"no actions are defined on {self.__class__.__name__}.")

        # prepare a factory queue, which we will use to create action instances
        factory_queue = _make_factory_queue(self.actions)
        # check that all provided kwargs are valid for at least one action factory
        self._check_action_parameters(factory_queue, kwargs)

        self.action_queue: list[RmAction] = [f.make_action(carrier=carrier, cfg=cfg, **kwargs) for f in factory_queue]
        """List of RM actions to be executed in order."""

    def run(self, sim: Simulation, days_prior: int) -> None:
        """Run all actions in the RM system's action queue.

        This will call all the actions in the action queue in order, passing
        the simulation and days prior to each action's run method. Each action
        should handle its own logic for whether it should execute based on the
        current simulation state and days prior.
        """
        for action in self.action_queue:
            action.run(sim, days_prior)

    def __call__(self, sim: Simulation, days_prior: int) -> None:
        """Run all actions in the RM system's action queue.

        This method allows the RM system instance to be called directly (i.e.,
        as a callback), which internally invokes the run method.  See `run()` for
        details.
        """
        self.run(sim, days_prior)

    @classmethod
    def get_name(cls) -> str:
        """Get the name of this RM system class."""
        n = getattr(cls, "_name", None)
        if n is None:
            n = cls.__name__
        return n

    def __repr__(self) -> str:
        return (
            f"<RmSys {self.get_name()!r}: {len(self.action_queue)} actions, "
            f"availability_control={self.availability_control}>"
        )


### RM SYSTEM REGISTRATION ###

_REGISTERED_SYSTEMS: dict[str, type[RmSys]] = {}


# Decorator to register a revenue management system class with a given name
def register_rm_system(rm_system: type[RmSys]) -> type[RmSys]:
    """Register an RM system class.

    This can be used as a decorator on an RmSys subclass.  By default,
    the name used for registration is the class name, unless the class
    defines a class variable `_name`, in which case that is used instead.
    The registered RM system can then be retrieved by name using
    `get_registered_rm_system(name)`. You can also use registered RM systems
    by name in configuration files, to attach them to carriers.

    Parameters
    ----------
    rm_system : type[RmSys]
        The RM system class to register.

    Returns
    -------
    type[RmSys]
        The same RM system class that was registered.  This makes it possible
        to use this function as a decorator to register RM system but otherwise
        act transparently.
    """
    global _REGISTERED_SYSTEMS
    name = rm_system.get_name()
    if name in _REGISTERED_SYSTEMS:
        raise ValueError(f"RM system {name!r} is already registered.")
    _REGISTERED_SYSTEMS[name] = rm_system
    return rm_system


def get_registered_rm_system(name: str) -> type[RmSys]:
    """Retrieve a registered RM system class by name."""
    global _REGISTERED_SYSTEMS
    if name not in _REGISTERED_SYSTEMS:
        raise KeyError(f"RM system {name!r} is not registered.")
    return _REGISTERED_SYSTEMS[name]


def check_registered_rm_system(name: str) -> bool:
    """Check if an RM system with the given name is registered."""
    global _REGISTERED_SYSTEMS
    return name in _REGISTERED_SYSTEMS


def list_registered_rm_systems() -> list[str]:
    """List the names of all registered RM systems."""
    global _REGISTERED_SYSTEMS
    return list(_REGISTERED_SYSTEMS.keys())


def export_registered_rm_systems() -> bytes:
    """Serialize and export the dictionary of registered RM systems.

    Returns
    -------
    bytes
        The serialized data containing the registered RM systems.
    """

    global _REGISTERED_SYSTEMS
    return dill.dumps(_REGISTERED_SYSTEMS)


def restore_registered_rm_systems(data: bytes) -> None:
    """Restore the dictionary of registered RM systems from serialized data.

    This will update the existing registered RM systems with those
    found in the provided data. Existing registrations with the same
    names will be overwritten, but others will be preserved.

    Parameters
    ----------
    data : bytes
        The serialized data containing the registered RM systems.
    """
    global _REGISTERED_SYSTEMS
    _REGISTERED_SYSTEMS.update(dill.loads(data))
