from typing import Any

from pydantic import BaseModel
from pydantic_core import SchemaValidator

from .base import Config


def manipulate_config(cfg: Config, key: str, value: Any) -> Config:
    """Manipulate a config object by setting a key to a value.

    Parameters
    ----------
    cfg : Config
        The config object to manipulate.
    key : str
        The key to set.
    value : Any
        The value to set the key to.

    Returns
    -------
    Config
        The manipulated config object.
    """
    k, *remainder = key.split(".")
    remainder = ".".join(remainder)
    if remainder:
        try:
            i = int(k)
        except ValueError:
            self = getattr(cfg, k)
        else:
            self = cfg[i]
        manipulate_config(self, remainder, value)
    else:
        try:
            i = int(k)
        except ValueError:
            setattr(cfg, k, value)
        else:
            cfg[i] = value
    return cfg


def revalidate(model: BaseModel) -> None:
    """Revalidate a model using its schema."""
    schema_validator = SchemaValidator(schema=model.__pydantic_core_schema__)
    schema_validator.validate_python(model.__dict__)


def strip_all_restrictions(cfg: Config, inplace: bool = False) -> Config:
    """Remove all restrictions (except APs) from a PassengerSim network.

    Parameters
    ----------
    cfg : Config
        The PassengerSim network to strip restrictions from.
    inplace : bool, optional
        Whether to modify the original network in place, by default False.
        Setting this value to True will modify the original network config, which
        may be faster than creating a copy, but may also be undesirable if you
        want to keep the original network available for any reason.

    Returns
    -------
    Config
        The PassengerSim network with all restrictions removed.
    """
    if not inplace:
        cfg = cfg.model_copy(deep=True)
    for cm in cfg.choice_models.values():
        cm.restrictions.clear()
    for fare in cfg.fares:
        fare.restrictions.clear()
    return cfg
