# TITLE: RM Systems
from __future__ import annotations

from typing import Literal

from pydantic import field_validator

from .named import Named
from .rm_steps import RmStepBase

RmStep = RmStepBase.as_pydantic_field()
RmProcess = list[RmStep]


class RmSystem(Named, extra="forbid"):
    processes: dict[str, RmProcess]

    availability_control: Literal["infer", "leg", "theft", "bp", "vn", "none"] = "infer"
    """Fare class availability algorithm for carriers using this RmSystem.

    The default value will infer the appropriate control based on the step_type
    of the last step in `processes['dcp']`. (This is pending implementation).

    Allowed values include:
    - "leg" (default): Uses leg-based controls.
    - "bp": Bid price controls.
    - "vn": Virtual nesting.
    - "none": No controls.
    """

    @field_validator("processes")
    @classmethod
    def _require_dcp_process(cls, value: dict[str, RmProcess]):
        """Ensure that every RmSystem is either empty or has a DCP process.

        This validator also converts all keys to lowercase.
        """
        lower_value = {k.lower(): v for (k, v) in value.items()}
        if len(lower_value) and "dcp" not in lower_value:
            raise ValueError("Non-empty RmSystem missing a `dcp` process.")
        return lower_value
