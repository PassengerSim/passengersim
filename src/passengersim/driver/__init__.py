"""Main simulation driver for PassengerSim.

Provides the :class:`~passengersim.driver.Simulation` and
:class:`~passengersim.driver.MultiSimulation` classes, which load a
configuration, run the simulation engine, and return summarized results.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys  # noqa: F401
from typing import TypeVar

from passengersim_core import Ancillary, ContextualOptimizer, CustomerModel

from passengersim.summaries.generic import GenericSimulationTables
from passengersim.utils.nested_dict import from_nested_dict  # noqa: F401
from passengersim.utils.si import si_units  # noqa: F401
from passengersim.utils.tempdir import MaybeTemporaryDirectory  # noqa: F401

from ._base_sim import BaseSimulation
from ._constructors import make_core_choice_model, make_core_leg
from ._demand_gen import allocate_sample_demands, generate_sample_demands
from ._firehose import Firehose
from ._multiproc import MultiSimulation
from ._singleproc import Simulation, check_summarizer, get_default_summarizer, memory_log, set_default_summarizer

logger = logging.getLogger("passengersim")

SimulationTablesT = TypeVar("SimulationTablesT", bound=GenericSimulationTables)

_warn_skips = (os.path.dirname(__file__), os.path.dirname(contextlib.__file__))
