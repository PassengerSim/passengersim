from __future__ import annotations

import importlib
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


def make_rm_system_variant(new_cls: type[RmSys]) -> type[RmSys]:
    """Define a named variant of an existing RM system, with particular options.

    Use this decorator on a class which defines alternative default values
    for RmSys options, and which inherits from an existing RmSys class.
    This will create a new RM system class with these defined values as the
    defaults, and register it with the name of the new class.

    Raises
    ------
    ValueError
        The an RM system class with the given name is already registered.

    Examples
    --------
    To start with the existing `Qu` system, but change the default fare adjustment
    algorithm to `ki` and the fare adjustment to 0.25, do

    >>> from passengersim.rm.standard_systems import Qu
    >>> @make_rm_system_variant
    ... class Qu25(Qu):
    ...     fare_adjustment = "ki"
    ...     fare_adjustment_scale = 0.25

    """

    # the new class must be defined with exactly one base class, so we can easily determine which kwargs are new
    if len(new_cls.__bases__) != 1:
        raise ValueError(
            f"A new variant RmSys must have exactly one base class, but {new_cls} has {len(new_cls.__bases__)}"
        )

    base_cls = new_cls.__bases__[0]

    variant_defines = {
        k: v
        for k, v in new_cls.__dict__.items()
        if not k.startswith("__")  # Exclude magic methods/docs
    }

    def _new_init(self, *args, **kwargs):
        for k, v in variant_defines.items():
            if k not in kwargs:
                kwargs[k] = v
        super(base_cls, self).__init__(*args, **kwargs)

    final_cls = type(
        new_cls.__name__,
        (base_cls,),
        {"__init__": _new_init, "_variant_defines": variant_defines, "_name": new_cls.__name__},
    )

    return register_rm_system(final_cls)


def describe_rm_systems(cfg: Config | None = None) -> dict:
    """Describe the RM systems used in the configuration.

    If no configuration is provided, a description of all available systems is returned.
    """
    rm_systems_setup = {}
    if cfg is None:
        for k in list_registered_rm_systems():
            s = get_registered_rm_system(k)
            if hasattr(s, "_variant_defines"):
                rm_systems_setup[k] = {
                    "base_class_name": s.__base__.get_name(),
                    "variant_defines": s._variant_defines,
                }
            else:
                v = {"module": s.__module__}
                class_name = s.get_name()
                if class_name != k:
                    v["class_name"] = class_name
                if s.__name__ != k:
                    v["import_name"] = s.__name__
                rm_systems_setup[k] = v
    else:
        for carrier in cfg.carriers.values():
            k = carrier.rm_system
            s = get_registered_rm_system(k)
            while hasattr(s, "_variant_defines"):
                rm_systems_setup[k] = {
                    "base_class_name": s.__base__.get_name(),
                    "variant_defines": s._variant_defines,
                }
                s = s.__base__
                k = s.get_name()
                if get_registered_rm_system(k) is not s:
                    s = None
            if s is not None:
                v = {"module": s.__module__}
                class_name = s.get_name()
                if class_name != k:
                    v["class_name"] = class_name
                if s.__name__ != k:
                    v["import_name"] = s.__name__
                rm_systems_setup[k] = v

    return rm_systems_setup


def reload_rm_systems(description: dict) -> None:
    """Reload RM systems used in the configuration, based on the provided description."""
    step1 = {}
    step2 = []
    for k, v in description.items():
        if "variant_defines" in v:
            step2.append([k, v])
        else:
            step1[k] = v
    for k, v in step1.items():
        _module = importlib.import_module(v["module"])
        _import_name = v.get("import_name", k)
        _obj = getattr(_module, _import_name)
        if get_registered_rm_system(k) is not _obj:
            raise ValueError(f"tried and failed to load {_import_name!r} as {k!r} from {v['module']!r}")
    n = 0
    while len(step2):
        (k, v) = step2.pop(0)
        try:
            base_cls = get_registered_rm_system(v["base_class_name"])
        except (ValueError, KeyError):
            step2.append([k, v])
            n += 1
        if n > 1000:
            raise RecursionError("Recursion limit reached")
        variant_defines = v["variant_defines"]

        # if this system is already registered, check that it matches
        try:
            existing_system = get_registered_rm_system(k)
        except KeyError:
            existing_system = None
        if existing_system is not None:
            if existing_system.__base__.get_name() != v["base_class_name"]:
                raise ValueError(f"existing system {k!r} has different base class than expected")
            if getattr(existing_system, "_variant_defines", None) != variant_defines:
                raise ValueError(f"existing system {k!r} has different settings than expected")
            continue

        # existing_system is None, so we need to create and register it
        _new_cls = make_rm_system_variant(type(k, (base_cls,), variant_defines))

        # we don't need to do anything else with _new_cls, it should now be registered
        if get_registered_rm_system(k) is not _new_cls:
            raise ValueError("unexpected error")
