import logging

from passengersim.config.base import Config, filterable_list

logger = logging.getLogger("passengersim.transforms")


def remove_place(cfg: Config, place: str, inplace: bool = False) -> Config:
    """Remove a place from the configuration.

    This will not only remove the place from the `places`, but also remove
    any legs, paths, markets, demands, and fares that touch the place.
    """
    if not inplace:
        cfg = cfg.model_copy(deep=True)

    if place in cfg.places:
        logger.info(f"Removing place {place}: {cfg.places[place]}")
        del cfg.places[place]

    removed_leg_ids = set()
    legs = filterable_list()
    for leg in cfg.legs:
        if leg.orig != place and leg.dest != place:
            legs.append(leg)
        else:
            removed_leg_ids.add(leg.leg_id)
    cfg.legs = legs

    paths = filterable_list()
    for path in cfg.paths:
        if any(leg_id in removed_leg_ids for leg_id in path.legs):
            continue
        paths.append(path)
    cfg.paths = paths

    markets = filterable_list()
    for market in cfg.markets:
        if market.orig != place and market.dest != place:
            markets.append(market)
    cfg.markets = markets

    demands = filterable_list()
    for demand in cfg.demands:
        if demand.orig != place and demand.dest != place:
            demands.append(demand)
    cfg.demands = demands

    fares = filterable_list()
    for fare in cfg.fares:
        if fare.orig != place and fare.dest != place:
            fares.append(fare)
    cfg.fares = fares

    return cfg
