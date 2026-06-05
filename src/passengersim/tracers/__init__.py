"""Tracers for recording detailed simulation data.

Tracers attach to a simulation, to collect and summarize fine-grained data — such as
individual leg bid prices and path-level demand forecasts — on every sample, making
it possible to more deeply analyze simulation progression and within-sample dynamics.
"""

from __future__ import annotations

from .bid_price import LegBidPriceTracer
from .forecasts import PathForecastTracer
