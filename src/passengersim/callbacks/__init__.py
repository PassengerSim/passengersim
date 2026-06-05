"""Callback hooks triggered at key points during a simulation.

Callbacks can be registered to fire at the beginning of each sample,
at the end of each day, or at the end of each sample, allowing custom
logic to run alongside the simulation engine without modifying the core.
"""

from .base import CallbackData, CallbackMixin
from .registration import register_begin_sample_callback, register_daily_callback, register_end_sample_callback
