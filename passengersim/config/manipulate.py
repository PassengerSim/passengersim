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
