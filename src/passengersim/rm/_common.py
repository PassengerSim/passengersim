from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import TypeAdapter

from ._system_options import RmSysOption

if TYPE_CHECKING:
    from passengersim.config import Config
    from passengersim.driver import Simulation
    from passengersim.snapshot import SnapshotInstruction
    from passengersim.snapshot.filtering import GenericSnapshotFilter


class RmActionFactory:
    def __init__(self, action_class: type[RmAction], fixed_values: dict[str, Any] | None = None, **config_kwargs):
        self.action_class = action_class
        """The RmAction subclass to instantiate."""

        self.config_kwargs: dict[str, RmSysOption] = {}
        """Keyword arguments to pass to the RmAction constructor.

        Each keyword argument name should correspond to a parameter in the
        RmAction subclass's __init__ method. The value of each keyword argument
        will be the keyword argument used in the `RmSys` that uses this
        factory, or a `RmSysOption` whose name is that keyword.
        """
        for k, v in config_kwargs.items():
            if isinstance(v, str):
                v = RmSysOption(name=v)
            if not isinstance(v, RmSysOption):
                raise TypeError(
                    f"Configurable parameter '{k}' in RmActionFactory for {self.action_class.__name__} "
                    f"must be a string or RmSysOption; found {type(v).__name__} instead."
                )
            self.config_kwargs[k] = v

        self.fixed_kwargs: dict[str, Any] = fixed_values or {}
        """Fixed keyword arguments to always pass to the RmAction constructor.

        This allows certain parameters of the underlying RmAction to be fixed
        to specific non-configurable values. Use fixed values to set values that
        are not the default for the RmAction, but should not be configurable
        via the RmSys.  For example, if a system should always apply untruncation
        on only yieldable demand (the default is to apply untruncation on all demand),
        then the `fixed_values` can be used to set that parameter in a way that
        users of this RmSystem won't be able to change it accidentally.
        """

        # check that there is no overlap between config_kwargs and fixed_values
        for k in self.fixed_kwargs:
            if k in self.config_kwargs:
                raise ValueError(
                    f"Cannot have both a fixed value and a configurable value for parameter '{k}' "
                    f"in RmActionFactory for {self.action_class.__name__}."
                )

    def make_action(self, carrier: str, cfg: Config | None = None, **kwargs) -> RmAction:
        """Create an instance of the RmAction."""
        kw = {"carrier": carrier, "cfg": cfg}
        for k, v in self.config_kwargs.items():
            looking_for = v.name
            if looking_for in kwargs:
                # The user has provided a value for this option, use it
                kw[k] = kwargs[looking_for]
            elif v.has_default():
                # the user has not provided a value, but there is a default
                # provided by the RmSys that might override the default
                # in the RmAction
                kw[k] = v.default
        for k, v in self.fixed_kwargs.items():
            kw[k] = v
        return self.action_class(**kw)

    def has_option(self, option_name: str) -> bool:
        """Check if this factory has a configuration option with the given name."""
        for opt in self.config_kwargs.values():
            if opt.name == option_name:
                return True
        return False


class RmAction(ABC):
    """
    A revenue management action.

    Each RmAction is configured to run on a specific carrier, and on a set of
    days prior to departure.  Actions can be scheduled to run on specific
    days prior to departure (DCPs), or daily, or on other frequencies.
    """

    requires: set[str] = set()
    produces: set[str] = set()

    frequency: Literal["dcp", "daily", "daily_pre_dep", "non_dcp", "begin_sample", "end_sample", "weekly"] = None
    """The frequency with which to run this action.

    This can be one of the following values:
    - "dcp": run only on the specified DCPs.
    - "daily": run every day.
    - "daily_pre_dep": run every day prior to departure (i.e., days_prior > 0)."
    - "non_dcp": run on days that are not in the specified DCPs.
    - "begin_sample": run only on the first DCP (i.e., the maximum days_prior).
    - "end_sample": run only on the day of departure (i.e., days_prior == 0).
    - "weekly": run every 7 days (i.e. when days_prior is a multiple of 7).

    The `run` method of RM actions is actually called every day (as it is implemented
    as a daily callback), but the first thing the `run` method should do is check
    whether it should actually execute on that day, using the `should_run` method,
    which uses this frequency setting to determine whether to proceed.
    """

    dcps: set[int]
    """Set of days prior to departure on which to run this action."""

    snapshot_filter_type: type[GenericSnapshotFilter] = None

    @abstractmethod
    def run(self, sim: Simulation, days_prior: int):
        """Execute the action for the given simulation.

        Subclasses must implement this method.
        """
        raise NotImplementedError

    @classmethod
    def configure(cls, fixed: dict[str, Any] | None = None, **kwargs) -> RmActionFactory:
        """Create an RmActionFactory for this action with the given configuration.

        Each keyword argument name should correspond to a parameter in the
        RmAction subclass's __init__ method. The value of each keyword argument
        will be the keyword argument used in the `RmSys` that uses this
        factory.

        Fixed values can be provided via the `fixed` parameter, which is a
        dictionary of parameter names to fixed values. These values will always
        be passed to the RmAction constructor, and cannot be overridden via
        the RmSys.
        """
        return RmActionFactory(cls, fixed_values=fixed, **kwargs)

    def __init__(
        self,
        *,
        carrier: str = "",
        minimum_sample: int = 10,
        cfg: Config | None = None,
        snapshot_filters: Sequence | None = None,
    ):
        self.carrier = carrier
        """The carrier upon which to apply this action."""

        self.minimum_sample = minimum_sample
        """The minimum sample number before this action will run."""

        if cfg is not None:
            self.dcps = set(cfg.dcps)
        else:
            self.dcps = set([])

        self._dcp_index = {j: i for i, j in enumerate(reversed(sorted(self.dcps)))}
        self._max_dcp = max(self.dcps) if self.dcps else 1

        # validation checks
        if not self.carrier:
            raise ValueError(f"Carrier must be specified for {self.__class__.__name__}.")
        if not self.dcps:
            raise ValueError(f"At least one DCP must be specified for {self.__class__.__name__}.")
        if self.frequency is None:
            raise ValueError(
                f"Frequency must be specified for {self.__class__.__name__} (usually as a class attribute)."
            )

        self._snapshot_filters: list = []
        """One or more snapshot filters to apply while running this action."""

        if snapshot_filters:
            if self.snapshot_filter_type is not None:
                # if snapshot_filters is not a sequence, convert it to one
                if not isinstance(snapshot_filters, Sequence):
                    snapshot_filters = [snapshot_filters]
                for sf in snapshot_filters:
                    if isinstance(sf, self.snapshot_filter_type):
                        self._snapshot_filters.append(sf)
                    elif isinstance(sf, dict):
                        self._snapshot_filters.append(self.snapshot_filter_type(**sf))
                    else:
                        raise TypeError(f"incompatible snapshot filter[s], requires {self.snapshot_filter_type}")
            else:
                raise ValueError(f"unexpected snapshot filter[s] on {self.__class__.__name__}")

    def apply_snapshot_filters(self, sim: Simulation, days_prior: int, *args, **kwargs) -> SnapshotInstruction | None:
        """Apply this action's snapshot filters, if any, and return the resulting instruction.

        If there are no snapshot filters, or if none of the filters trigger, then this returns None.
        """
        if self._snapshot_filters:
            for snap_filter in self._snapshot_filters:
                instruction = snap_filter.resolve(sim.eng, days_prior, *args, **kwargs)
                if instruction:
                    print(f"triggered snapshot filter: {snap_filter}, instruction: {instruction}")
                    return instruction
        return None

    def get_dcp_index(self, days_prior: int, allow_between: bool = False) -> int:
        try:
            return self._dcp_index[days_prior]
        except KeyError:
            if not allow_between:
                raise ValueError(f"days_prior {days_prior} is not a DCP for {self.__class__.__name__}.") from None
            if days_prior > self._max_dcp:
                raise ValueError(
                    f"days_prior {days_prior} is before the initial DCP for {self.__class__.__name__}."
                ) from None
            if days_prior < 0:
                raise ValueError(f"days_prior {days_prior} is not valid for {self.__class__.__name__}.") from None
            while days_prior not in self.dcps and days_prior < self._max_dcp:
                days_prior += 1
            return self._dcp_index.get(days_prior, 0)

    def should_run(self, sim: Simulation, days_prior: int) -> bool:
        """Determine if the action should run on the given days_prior."""
        if sim.eng.sample < self.minimum_sample:
            return False
        match self.frequency:
            case "daily":
                return True
            case "daily_pre_dep":
                return days_prior > 0
            case "dcp":
                return days_prior in self.dcps and days_prior > 0
            case "non_dcp":
                return days_prior not in self.dcps and days_prior > 0
            case "begin_sample":
                return days_prior == self._max_dcp
            case "end_sample":
                return days_prior == 0
            case "weekly":
                return days_prior % 7 == 0
            case _:
                return False


class RmActionCfg(RmAction):
    """A configurable RM action.

    This is a base class for RM actions that have a configuration that can be
    loaded from the RM system's other_controls dictionary in the simulation
    configuration.  This allows arbitrarily complex RM action configurations to
    be defined, loaded, and validated from the simulation configuration, and for
    multiple carriers to easily share the same complex configuration.
    """

    cfg_name: str
    """The name of this action's configuration option in the RM system's other_controls.

    This RM action will load its configuration from the top-level simulation
    configuration's `other_controls` dictionary, using this name as the key.
    """

    cfg_type: type
    """The type of this action's configuration option.

    This type will be used to validate and parse the raw configuration
    value from the RM system's other_controls.
    """

    __adapter = None

    @classmethod
    def _type_adapter(cls):
        """Get the TypeAdapter for this action's configuration type.

        The adapter is used to validate the RM action's configuration inputs,
        when they are loaded from the Config object. This is cached for efficiency.
        """
        if cls.__adapter is None:
            cls.__adapter = TypeAdapter(cls.cfg_type)
        return cls.__adapter

    def __init__(
        self,
        *,
        carrier: str,
        minimum_sample: int = 10,
        cfg: Config | None = None,
    ):
        super().__init__(carrier=carrier, minimum_sample=minimum_sample, cfg=cfg)
        if cfg is None:
            raw = None
        else:
            raw = cfg.other_controls.get(self.cfg_name)

        self.action_cfg = self._type_adapter().validate_python(raw)
        """Parsed and validated configuration for this RM action.

        This should be an instance of the type specified by the class `cfg_type`.
        """
