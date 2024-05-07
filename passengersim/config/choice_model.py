# TITLE: Choice Models
from __future__ import annotations

from typing import Annotated, Literal, List

from pydantic import BaseModel, Field

from .named import Named, DictOfNamed


class DwmData(BaseModel, extra="forbid"):
    min_distance: int = 0
    max_distance: int = 25000
    k_factor: float = 3.0
    earlyDepMultiplier: float = 10.0
    lateDepMultiplier: float = 10.0
    earlyArrMultiplier: float = 10.0
    lateArrMultiplier: float = 10.0
    probabilities: List[float] = []


class PodsChoiceModel(Named, extra="forbid"):
    kind: Literal["pods"]

    emult: float | None = None

    basefare_mult: float | None = None
    connect_disutility: float | None = None
    path_quality: tuple[float, float] | None = None
    preferred_airline: tuple[float, float] | None = None
    hhi: tuple[float, float] | None = None
    elapsed_time: tuple[float, float] | None = None
    buffer_threshold: int | None = None
    buffer_time: tuple[float, float] | None = None
    r1: float | None = None
    r2: float | None = None
    r3: float | None = None
    r4: float | None = None
    dwm_data: List[DwmData] = []
    tolerance: float | None = None
    non_stop_multiplier: float | None = None
    connection_multiplier: float | None = None
    todd_curve: str | None = None

    anc1_relevance: float | None = None
    anc2_relevance: float | None = None
    anc3_relevance: float | None = None
    anc4_relevance: float | None = None


class LogitChoiceModel(Named, extra="forbid"):
    kind: Literal["logit"]

    emult: float | None = None
    """Using for WTP, a bit of a quick and dirty until we have a better approach"""

    r1: float | None = None
    r2: float | None = None
    r3: float | None = None
    r4: float | None = None

    anc1_relevance: float | None = None
    anc2_relevance: float | None = None
    anc3_relevance: float | None = None
    anc4_relevance: float | None = None

    intercept: float = 0
    """This is the alternative specific constant for the no-purchase alternative."""

    nonstop: float = 0
    duration: float = 0
    price: float = 0
    """This is the parameter for the price of each alternative."""

    tod_sin2p: float = 0
    r"""Schedule parameter.

    If $t$ is departure time (in minutes after midnight local time) divided
    by 1440, this parameter is multiplied by $sin( 2 \pi t)$ and the result is
    added to the utility of the particular alternative."""

    tod_sin4p: float = 0
    r"""Schedule parameter.

    If $t$ is departure time (in minutes after midnight local time) divided
    by 1440, this parameter is multiplied by $sin( 4 \pi t)$ and the result is
    added to the utility of the particular alternative."""

    tod_sin6p: float = 0
    tod_cos2p: float = 0
    tod_cos4p: float = 0
    tod_cos6p: float = 0
    free_bag: float = 0
    early_boarding: float = 0
    same_day_change: float = 0


ChoiceModel = Annotated[PodsChoiceModel | LogitChoiceModel, Field(discriminator="kind")]
"""
Two types of choice models are available in PassengerSim.

Use the `kind` key to select which kind of choice model you wish to parameterize.
"""
