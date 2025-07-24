from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from passengersim import Config


def drop_carriers(cfg: Config, carriers: list[str]) -> Config:
    """
    Drop carrier(s) from this config, including all fares, legs, and paths.

    Parameters
    ----------
    cfg : Config
        The configuration object containing fares, classes, and carriers.
    carriers : list of str
        The list of carrier codes to drop from the configuration.

    Returns
    -------
    Config
        The updated configuration object without the indicated carriers.
    """
    for c in carriers:
        # remove the carrier from the configuration
        if c in cfg.carriers:
            del cfg.carriers[c]
        # Remove fares for the dropped carrier
        fares = []
        for f in cfg.fares:
            if f.carrier != c:
                fares.append(f)
        cfg.fares = fares
        # Remove legs for the dropped carrier
        legs = []
        removed_leg_ids = set()
        for leg in cfg.legs:
            if leg.carrier != c:
                legs.append(leg)
            else:
                removed_leg_ids.add(leg.leg_id)
        cfg.legs = legs
        # Remove the carrier paths
        paths = []
        for p in cfg.paths:
            if p.legs and p.legs[0] not in removed_leg_ids:
                paths.append(p)
        cfg.paths = paths

    return cfg
