from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from passengersim import Config


def demand_multiplier(cfg: Config, multiplier: float | None = None) -> Config:
    """
    Scale the demand by a given multiplier.

    Parameters
    ----------
    cfg : Config
        The configuration object containing demands.
    multiplier : float, optional
        The multiplier to apply to the base demand. If None, scaling is taken from
        the configuration's `simulation_controls.demand_multiplier`.

    Returns
    -------
    Config
        The configuration object with scaled demands.
    """
    if multiplier is None:
        multiplier = cfg.simulation_controls.demand_multiplier
        if multiplier == 1.0:
            warnings.warn(
                "No demand multiplier specified either manually or in "
                "simulation_controls, so no scaling will be applied.",
                stacklevel=2,
            )
            return cfg
    else:
        if cfg.simulation_controls.demand_multiplier != 1.0:
            raise ValueError("Cannot specify a demand multiplier manually when one is set in simulation_controls.")

    for d in cfg.demands:
        d.base_demand *= multiplier

    # When complete, clear the demand multiplier in simulation_controls
    cfg.simulation_controls.demand_multiplier = 1.0
    return cfg
