from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from passengersim import Config


def pricing_multiplier(cfg: Config, multiplier: float) -> Config:
    """
    Scale fare prices and demand reference prices by a given multiplier.

    Parameters
    ----------
    cfg : Config
        The configuration object containing demands.
    multiplier : float, optional
        The multiplier to apply to the pricing.

    Returns
    -------
    Config
        The configuration object with scaled prices.
    """

    for d in cfg.demands:
        d.reference_price *= multiplier
    for f in cfg.fares:
        f.price *= multiplier
    return cfg
