"""Revenue management systems and algorithms for PassengerSim.

Provides a collection of built-in RM systems (EMSR, leg DP, Q-forecasting, etc.),
and the tools needed to register and configure custom revenue management strategies.
"""

from . import standard_systems
from ._common import RmAction
from .systems import (
    RmSys,
    RmSysOption,
    check_registered_rm_system,
    describe_rm_systems,
    get_registered_rm_system,
    list_registered_rm_systems,
    make_rm_system_variant,
    register_rm_system,
    reload_rm_systems,
)
