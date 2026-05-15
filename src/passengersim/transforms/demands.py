from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from passengersim import Config


def demand_multiplier(cfg: Config, multiplier: float | None = None, *, inplace: bool = False) -> Config:
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
    if not inplace:
        cfg = cfg.model_copy(deep=True)
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


def _has_common_reference_prices(
    cfg: Config,
):
    """Check if all demands that share a market also share a common reference price."""
    market_reference_prices = {}
    for d in cfg.demands:
        if d.market_identifier not in market_reference_prices:
            market_reference_prices[d.market_identifier] = d.reference_price
        else:
            if market_reference_prices[d.market_identifier] != d.reference_price:
                return False
    return True


def common_reference_prices(cfg: Config, anchor_segment: str, *, inplace: bool = False) -> Config:
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
    if not inplace:
        cfg = cfg.model_copy(deep=True)

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
                if _has_common_reference_prices(cfg):
                    # we are already done, no need to complain here
                    return cfg
                raise ValueError(f"choice model {cm.name} has existing reference_price_multiplier")

    # Set the new multiplier on each choice model
    for cm_name, cm in cfg.choice_models.items():
        cm.reference_price_multiplier = segment_multipliers[cm_name]

    # Set the market anchor reference prices onto all demands
    for d in cfg.demands:
        d.reference_price = market_reference_prices[d.market_identifier]

    return cfg


def dissolve_reference_price_multipliers(cfg: Config, *, inplace: bool = False) -> Config:
    """Remove reference price multipliers from all choice models.

    This pushes the effects of the existing multipliers into the reference prices
    set on the demands.
    """
    if not inplace:
        cfg = cfg.model_copy(deep=True)
    for d in cfg.demands:
        d.reference_price *= cfg.choice_models[d.choice_model or d.segment].reference_price_multiplier
    for cm in cfg.choice_models.values():
        cm.reference_price_multiplier = 1.0
    return cfg


def set_demand_reference_prices(
    cfg: Config,
    mult_on_lowest_price: float = 1.0,
    *,
    inplace: bool = False,
) -> Config:
    """Set reference prices on all demands equal to a multiple of the lowest price in the market.

    Parameters
    ----------
    cfg : Config
    mult_on_lowest_price : float

    Returns
    -------
    Config

    Raises
    ------
    ValueError
        If the existing reference prices are not consistent across segments.
    """
    if not inplace:
        cfg = cfg.model_copy(deep=True)

    from passengersim.config.checks.demand import check_reference_price_scaling

    df = check_reference_price_scaling(cfg)
    # this will have thrown an error if the reference prices are not consistent
    # across segments, so if we get here we know they are ok
    for d in cfg.demands:
        d.reference_price = mult_on_lowest_price * df.loc[d.market_identifier, "min_price"]
    return cfg


def quick_mean_wtp(emult, ref_price):
    return (emult * 1.44269940 - 0.44269940) * ref_price


def set_demand_mean_wtp(cfg: Config, mean_wtp: dict[str, float]):
    """Set demand reference prices so that they result in particular average maximum WTP."""
    for d in cfg.demands:
        # look for targets both in the forward and backward directions.
        # prefer the exact forward match if available, use backward otherwise
        identifier = d.identifier
        identifier_swap = f"{d.dest}~{d.orig}@{d.segment}"
        if identifier_swap in mean_wtp and identifier not in mean_wtp:
            identifier = identifier_swap
        if identifier not in mean_wtp:
            continue
        target = mean_wtp[identifier]
        cm_name = d.choice_model or d.segment
        cm = cfg.choice_models[cm_name]
        if d.emult is None:
            emult = cm.emult
        else:
            emult = d.emult
        reference_price_multiplier = cm.reference_price_multiplier
        ref_price = target / (emult * 1.44269940 - 0.44269940) / reference_price_multiplier
        d.reference_price = ref_price


def relevel_demand(cfg: Config, *, inplace: bool = False) -> Config:
    """Change reference prices to be greater than or equal to the minimum price in each market.

    This function will modify reference price, emult, and base demand levels so that the
    reference price is greater than or equal to the minimum price in each market, but the
    simulated demand remains the same except that demand with max WTP below the minimum price
    is never generated.
    """
    if not inplace:
        cfg = cfg.model_copy(deep=True)

    from passengersim.config.checks.markets import check_min_fare_price_by_market
    from passengersim.driver import make_core_choice_model

    min_prices_by_market: dict[str, float] = {k: v["min_price"] for k, v in check_min_fare_price_by_market(cfg).items()}

    # market_reference_prices = _get_market_reference_prices(cfg, anchor_segment)
    #
    for d in cfg.demands:
        min_price = min_prices_by_market[d.market_identifier]
        if min_price <= d.reference_price:
            # reference price is already above minimum price, do nothing
            continue
        cm_name = d.choice_model_
        cm = cfg.choice_models[cm_name]
        scale_up = min_price / d.reference_price
        emult = d.emult if d.emult is not None else getattr(cm, "emult", 1.5)
        new_emult = (emult - 1) / scale_up + 1
        core_cm = make_core_choice_model(cm)
        scale_demand = core_cm.prob_wtp(min_price, reference_price=d.reference_price, emult=emult)
        if scale_demand > 1.0:
            raise ValueError(
                f"unexpected {scale_demand=}>1, {emult=}, {min_price=}, reference_price={d.reference_price=}"
            )
        d.emult = new_emult
        d.reference_price = min_price
        d.base_demand *= scale_demand

    return cfg
