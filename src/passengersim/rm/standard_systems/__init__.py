"""Standard revenue management (RM) systems.

RM systems in this subpackage are available for general use, and are automatically
available for use with all PassengerSim models. You do not need to explicitly import
the relevant submodule to register them.
"""

from ._E import E
from ._L import L
from ._M import M
from ._P import P
from ._Q import Q, Qe, Qu
from ._U import U
from ._V import V
