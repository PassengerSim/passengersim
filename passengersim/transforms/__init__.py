"""
This module provides various transformations for PassengerSim configurations.

Transformations can be used to modify or clean the configuration data, so
that PassengerSim can run simulations more efficiently or display results
more intuitively.
"""

from .booking_classes import class_rename
from .carriers import drop_carriers
from .demands import demand_multiplier
from .restrictions import clean_restrictions
