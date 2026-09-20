from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from passengersim import Config


def clean_restrictions(cfg: Config, *, inplace: bool = False) -> Config:
    """
    Remove unused restrictions from choice models and fares.

    This function removes any restrictions from choice models that are not
    present in any fare, as they are basically superfluous. It also removes
    any restrictions from fares that are not present in any choice model, for
    the same reason. This helps to keep the configuration clean reduces memory
    usage by not storing unnecessary restrictions. Restriction names are
    matched case-insensitively, while their original spelling is preserved.

    Parameters
    ----------
    cfg : Config
        The configuration object containing fares and choice models.

    Returns
    -------
    Config
        The cleaned configuration object with unused restrictions removed.
    """
    if not inplace:
        cfg = cfg.model_copy(deep=True)

    fare_restrictions = {r.casefold() for fare in cfg.fares for r in fare.restrictions}
    choice_model_restrictions = {
        r.casefold() for choice_model in cfg.choice_models.values() for r in (choice_model.restrictions or {})
    }
    shared_restrictions = fare_restrictions & choice_model_restrictions

    for choice_model in cfg.choice_models.values():
        if choice_model.restrictions:
            choice_model.restrictions = {
                name: value
                for name, value in choice_model.restrictions.items()
                if name.casefold() in shared_restrictions
            }

    for fare in cfg.fares:
        fare.restrictions = [r for r in fare.restrictions if r.casefold() in shared_restrictions]

    return cfg
