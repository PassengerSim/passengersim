# TITLE: RM Steps
from __future__ import annotations

import inspect
import operator
import typing
from functools import reduce
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, Field


class RmStepBase(BaseModel, extra="forbid"):
    """
    Base class for RmSteps.

    Every non-abstract derived class must have a `step_type` class attribute,
    annotated with a `Literal` value type.

    See `DummyStep` below for an example.
    """

    __subclasses: ClassVar[set[type[RmStepBase]]] = set()

    @classmethod
    def _get_subclasses(cls) -> set[type[RmStepBase]]:
        """Return a set of all concrete subclasses"""
        return cls.__subclasses.copy()

    def __init_subclass__(cls, **kwargs):
        """Capture a list of all concrete subclasses, including nested levels"""
        super().__init_subclass__(**kwargs)

        if inspect.isabstract(cls):
            return  # do not consider intermediate abstract base classes

        annotations = inspect.get_annotations(cls, eval_str=True)
        assert "step_type" in annotations, "step_type not in annotations"
        annotation_step_type = typing.get_origin(annotations["step_type"])
        assert annotation_step_type == Literal, (
            f"annotation {annotations['step_type']} for `{cls.__name__}.step_type` "
            f"is not Literal but {annotation_step_type}"
        )
        found_step_type = typing.get_args(annotations["step_type"])[0]
        if cls.__name__.lower().endswith("step"):
            assert found_step_type == cls.__name__.lower()[:-4], (
                f"annotation Literal value {found_step_type!r} "
                f"for `{cls.__name__}.step_type` is not the same as the class name "
                f"(omitting 'step' suffix)"
            )
        else:
            assert found_step_type == cls.__name__.lower(), (
                f"annotation Literal value {found_step_type!r} "
                f"for `{cls.__name__}.step_type` is not the same as the class name"
            )
        cls.__subclasses.add(cls)

    @classmethod
    def as_pydantic_field(cls):
        """Pydantic field as a union of all subclasses, discriminated on step_type."""
        if len(cls.__subclasses) > 1:
            return Annotated[
                reduce(operator.__or__, cls.__subclasses),
                Field(discriminator="step_type"),
            ]
        else:  # only the DummyStep
            return Annotated[reduce(operator.__or__, cls.__subclasses), Field()]

    def _factory(self):
        """Generate a deep copy of this RmStep."""
        return self.model_copy(deep=True)

    def use_config(self, config):
        """Some RM Steps might want to check parts of the config."""
        pass


class DummyStep(RmStepBase):
    step_type: Literal["dummy"]
