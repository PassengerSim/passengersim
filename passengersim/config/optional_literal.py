from __future__ import annotations

from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, BeforeValidator, ValidationError

T = TypeVar("T")


def _interpret_none(v: T) -> T | None:
    """Convert an empty string or a similar literal to None."""
    if isinstance(v, str):
        if v == "":
            return None
        if v.lower() == "none":
            return None
        if v.lower() == "null":
            return None
        if v.lower() == "undefined":
            return None
        if v.lower() == "nil":
            return None
        if v.lower() == "no":
            return None
        if v.lower() == "off":
            return None
    return v


# This "Optional" type is used to allow None or a specific type T.
# It uses a BeforeValidator to interpret certain string values as None,
# such as an empty string, "None", "null" or "off".
Optional = Annotated[T | None, BeforeValidator(_interpret_none)]


if __name__ == "__main__":
    # Example usage
    class ExampleModel(BaseModel):
        optional_field: Optional[Literal["example", "test"]] = None

    try:
        example = ExampleModel(optional_field="noppp")
    except ValidationError as e:
        print(e)
    ee = ExampleModel(optional_field="example")
    print(ee.optional_field)  # Output: example
    ee = ExampleModel(optional_field="test")
    print(ee.optional_field)  # Output: test
    ee = ExampleModel(optional_field="")
    print(ee.optional_field)  # Output: None
    ee = ExampleModel(optional_field="no")
    print(ee.optional_field)  # Output: None
    ee = ExampleModel(optional_field="NoNe")
    print(ee.optional_field)  # Output: None
