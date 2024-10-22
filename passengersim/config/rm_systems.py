# TITLE: RM Systems
from __future__ import annotations

from typing import Literal

from pydantic import field_validator

from .named import DictAttr, ListOfNamed, Named
from .rm_steps import RmStepBase

RmStep = RmStepBase.as_pydantic_field()
RmProcess = ListOfNamed[RmStep]


class RmSystem(Named, extra="forbid"):
    processes: DictAttr[str, RmProcess]

    availability_control: Literal[
        "infer", "leg", "cabin", "theft", "bp", "bp_loose", "vn", "none"
    ] = "infer"
    """Fare class availability algorithm for carriers using this RmSystem.

    The default value will infer the appropriate control based on the steps in
    the DCP process (This is pending implementation).

    Allowed values include:
    - "leg" (default): Uses leg-based controls.
    - "bp": Bid price controls with strict resolution (fare must be strictly
            greater than bid price).
    - "bp_loose": Bid price controls with non-strict resolution (fare must be
                  greater than *or equal to* bid price).
    - "vn": Virtual nesting.
    - "none": No controls.
    """

    @field_validator("processes")
    @classmethod
    def _require_dcp_process(cls, value: dict[str, RmProcess]):
        """Ensure that every RmSystem is either empty or has a DCP process.

        This validator also converts all keys to lowercase.
        """
        lower_value = DictAttr()
        for k, v in value.items():
            lower_value[k.lower()] = v
        if len(lower_value) and "dcp" not in lower_value:
            raise ValueError("Non-empty RmSystem missing a `dcp` process.")
        return lower_value
