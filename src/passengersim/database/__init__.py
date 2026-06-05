"""Database interface for persisting and querying simulation results.

Provides the :class:`~passengersim.database.Database` class and a set of
common queries for retrieving summary statistics, leg-level data, and other
outputs stored during or after a simulation run.
"""

from . import common_queries, tables
from .database import Database
