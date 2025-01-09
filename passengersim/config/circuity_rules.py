#
# Specifies exceptions and the default circuity rule for conneciton generation
#
# AlanW, October 2024
# (c) PassengerSim LLC
#

from __future__ import annotations

from .named import Named


class CircuityRule(Named, extra="forbid"):
    carrier: str | None = None
    orig_airport: str | None = None
    connect_airport: str | None = None
    dest_airport: str | None = None
    orig_state: str | None = None
    dest_state: str | None = None

    # The max circuity will be:  alpha + beta * market_distance
    # To make it unlimited, set a really high beta value, like 1000.0
    # To prohibit a conection, set alpha and beta to 0.0
    alpha: float
    beta: float
