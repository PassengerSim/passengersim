from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

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


def _get_market_reference_prices(
    cfg: Config,
    anchor_segment: str,
) -> dict[str, float]:
    market_reference_prices = {}

    # establish the market anchor reference price for all markets
    for d in cfg.demands:
        cm = d.choice_model or d.segment
        if cm == anchor_segment:
            if d.market_identifier in market_reference_prices:
                if market_reference_prices[d.market_identifier] != d.reference_price:
                    raise ValueError("inconsistent market reference price anchors")
            market_reference_prices[d.market_identifier] = d.reference_price

    return market_reference_prices


def common_reference_prices(
    cfg: Config,
    anchor_segment: str,
) -> Config:
    """
    Change from setting unique reference prices on demands to scaling them on choice models.

    This transform presumes all choice models are PODS choice models.  It will fail if
    reference prices are not currently consistently scaled.

    Parameters
    ----------
    cfg : Config
    anchor_segment : str
        The name of the passenger segment where the current reference prices are
        what the resulting reference prices should be (i.e. this choice model will
        have a reference_price_multiplier of 1.0).

    Returns
    -------
    Config
    """
    # establish the market anchor reference price for all markets
    market_reference_prices = _get_market_reference_prices(cfg, anchor_segment)

    # check that all demands have a market anchor reference price
    # and that by segment they are all the same multiple of that anchor
    segment_multipliers = {}
    for d in cfg.demands:
        if d.market_identifier not in market_reference_prices:
            raise ValueError(f"missing market anchor reference price on {d.market_identifier}")
        cm = d.choice_model or d.segment
        if cm in segment_multipliers:
            new_mult = d.reference_price / market_reference_prices[d.market_identifier]
            if not np.isclose(new_mult, segment_multipliers[cm]):
                raise ValueError(f"inconsistent segment multipliers for {d.market_identifier}")
        else:
            segment_multipliers[cm] = d.reference_price / market_reference_prices[d.market_identifier]

    # check that all existing choice models do not use reference_price_multiplier
    for cm in cfg.choice_models.values():
        if hasattr(cm, "reference_price_multiplier"):
            if cm.reference_price_multiplier is not None and cm.reference_price_multiplier != 1.0:
                raise ValueError(f"choice model {cm.name} has existing reference_price_multiplier")

    # Set the new multiplier on each choice model
    for cm_name, cm in cfg.choice_models.items():
        cm.reference_price_multiplier = segment_multipliers[cm_name]

    # Set the market anchor reference prices onto all demands
    for d in cfg.demands:
        d.reference_price = market_reference_prices[d.market_identifier]

    return cfg
