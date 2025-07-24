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

    availability_control: Literal["infer", "leg", "cabin", "theft", "bp", "bp_loose", "vn", "none"] = "infer"
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

    description: str = ""
    """Description of the RM system.

    The description is optional and can be used to summarize the RM system.
    It has no effect on the actual operation of the RM system."""

    frat5: str | None = None
    """Default Frat5 curve to use for the carrier if not otherwise defined.

    Some RM systems strictly require a Frat5 curve to be defined for every
    carrier and market. This attribute allows the user to define a default
    Frat5 curve to be used as the global default by any carrier assigned this
    RM system. If a carrier defines its own global `frat5` then that value will
    override this default.  If this attribute is set to `None`, then the RM system
    does not define a default Frat5 curve, and the carrier must define its own
    `frat5` attribute if necessary. Some RM systems do not require a Frat5 curve
    to be defined, in which case this attribute can be left as `None` without
    affecting the operation of the RM system.
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
