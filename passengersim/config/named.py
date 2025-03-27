"""This module contains utilities for managing named objects."""

from __future__ import annotations

import types
from typing import Annotated, Any, TypeVar

import addicty
from pydantic import GetCoreSchemaHandler
from pydantic.functional_validators import BeforeValidator
from pydantic_core import CoreSchema, core_schema

from .pretty import PrettyModel


class Dict(addicty.Dict):
    def __repr__(self):
        return dict.__repr__(self)

    @classmethod
    def __get_pydantic_core_schema__(
        cls: Any, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        if (
            isinstance(source_type, types.GenericAlias)
            and source_type.__origin__ is Dict
        ):
            return core_schema.no_info_after_validator_function(
                cls, handler(dict[source_type.__args__])
            )
        else:
            return core_schema.no_info_after_validator_function(
                cls, handler(source_type)
            )


class DictAttr(dict):
    def __getattr__(self, item):
        if item in self:
            return self[item]
        if item.lower() in self:
            return self[item.lower()]
        raise AttributeError(f"no key {item}")

    def __setattr__(self, item, value):
        self[item.lower()] = value

    @classmethod
    def __get_pydantic_core_schema__(
        cls: Any, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        if (
            isinstance(source_type, types.GenericAlias)
            and source_type.__origin__ is DictAttr
        ):
            return core_schema.no_info_after_validator_function(
                cls, handler(dict[source_type.__args__])
            )
        else:
            return core_schema.no_info_after_validator_function(
                cls, handler(source_type)
            )


class Named(PrettyModel):
    name: str


T = TypeVar("T", bound=Named)


def enforce_name(x: dict[str, T] | list[T]) -> dict[str, T]:
    """Enforce that each item has a unique name.

    If you provide a list, this will ensure that each item in the list has a name.
    If you provide a dict, the names are given by the keys of the dictionary, and
    this will ensure that for each value, if it also has an explicitly defined name,
    that name matches its key-derived name.
    """
    if isinstance(x, list):
        x_ = {}
        for n, i in enumerate(x):
            k = i.get("name")
            if k is None:
                raise ValueError(f"missing name in position {n}")
            x_[k] = i
        x = x_
    for k, v in x.items():
        try:
            missing_name = "name" not in v or not v["name"]
        except TypeError:
            missing_name = True
        if missing_name:
            try:
                v["name"] = k
            except TypeError:
                try:
                    v.name = k
                except AttributeError:
                    raise ValueError(f"cannot assign name {k!r} to {type(v)}") from None
        try:
            if v["name"] != k:
                raise ValueError(f"explict name {v['name']!r} does not match key {k!r}")
        except TypeError:
            if v.name != k:
                raise ValueError(
                    f"explict name {v.name!r} does not match key {k!r}"
                ) from None
    return x


DictOfNamed = Annotated[DictAttr[str, T], BeforeValidator(enforce_name)]


class ListOfNamed(list):
    @classmethod
    def __get_pydantic_core_schema__(
        cls: Any, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        if (
            isinstance(source_type, types.GenericAlias)
            and source_type.__origin__ is ListOfNamed
        ):
            return core_schema.no_info_after_validator_function(
                cls, handler(list[source_type.__args__])
            )
        else:
            return core_schema.no_info_after_validator_function(
                cls, handler(source_type)
            )

    def __getattr__(self, item):
        for step in self:
            if getattr(step, "step_type", None) == item:
                return step
        for step in self:
            if getattr(step, "name", None) == item:
                return step
        raise AttributeError(f"no step with step_type or name {item}")

    def __delattr__(self, item):
        for i, step in enumerate(self):
            if getattr(step, "step_type", None) == item:
                del self[i]
                return
        for i, step in enumerate(self):
            if getattr(step, "name", None) == item:
                del self[i]
                return
        raise AttributeError(f"no step with step_type or name {item}")
