from __future__ import annotations

from .base import CallbackMixin

# Registered callbacks are subsequently available to all simulations via the `use_registered_callbacks` method
_REGISTERED_CALLBACKS = CallbackMixin()

# generic decorators to register callbacks for later use.
register_daily_callback = _REGISTERED_CALLBACKS.daily_callback
register_begin_sample_callback = _REGISTERED_CALLBACKS.begin_sample_callback
register_end_sample_callback = _REGISTERED_CALLBACKS.end_sample_callback
