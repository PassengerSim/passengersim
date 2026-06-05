"""Tools for building itinerary connections between flight legs.

Connections represent valid multi-leg itineraries that passengers can book.
This module provides utilities to construct, validate, and pre-build those
connections based on schedule timing and circuity constraints.
"""

from passengersim_core._Zoo import build_connections

from . import circuity
from .checking import check_connections
from .prebuild import prebuild_connections

__all__ = ["build_connections", "circuity", "check_connections", "prebuild_connections"]
