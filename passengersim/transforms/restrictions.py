from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from passengersim import Config


def clean_restrictions(cfg: Config) -> Config:
    """
    Remove unused restrictions from choice models and fares.

    This function removes any restrictions from choice models that are not
    present in any fare, as they are basically superfluous. It also removes
    any restrictions from fares that are not present in any choice model, for
    the same reason. This helps to keep the configuration clean reduces memory
    usage by not storing unnecessary restrictions.

    Parameters
    ----------
    cfg : Config
        The configuration object containing fares and choice models.

    Returns
    -------
    Config
        The cleaned configuration object with unused restrictions removed.
    """
    all_restrictions_in_fares = set()
    for f in cfg.fares:
        for r in f.restrictions:
            all_restrictions_in_fares.add(r)

    all_restrictions_in_choicemodels = set()
    for cm in cfg.choice_models.values():
        for r in cm.restrictions:
            all_restrictions_in_choicemodels.add(r)

    for r in all_restrictions_in_choicemodels - all_restrictions_in_fares:
        for cm in cfg.choice_models.values():
            if r in cm.restrictions:
                del cm.restrictions[r]

    for r in all_restrictions_in_fares - all_restrictions_in_choicemodels:
        for f in cfg.fares:
            if r in f.restrictions:
                f.restrictions.remove(r)

    return cfg
