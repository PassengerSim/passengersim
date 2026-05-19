from __future__ import annotations

from passengersim.config import Config

from .demands import demand_multiplier


def resolve_market_demand_multipliers(cfg: Config, *, inplace: bool = True) -> Config:
    """
    Move the effect of all market demand multipliers to the demands.

    Parameters
    ----------
    cfg : Config
        The configuration object containing markets and demands.

    Returns
    -------
    Config
        The configuration object with scaled demands.
    """
    if not inplace:
        cfg = cfg.model_copy(deep=True)
    for mkt in cfg.markets:
        if mkt.demand_multiplier != 1.0:
            for d in cfg.demands:
                if d.market_identifier == mkt.identifier:
                    d.base_demand *= mkt.demand_multiplier
            mkt.demand_multiplier = 1.0
    return cfg


def simplify_config(cfg: Config, *, inplace: bool = True) -> Config:
    if not inplace:
        cfg = cfg.model_copy(deep=True)
    resolve_market_demand_multipliers(cfg, inplace=True)
    if cfg.simulation_controls.demand_multiplier != 1.0:
        demand_multiplier(cfg, inplace=True)
    return cfg
