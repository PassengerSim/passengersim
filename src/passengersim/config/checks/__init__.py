from __future__ import annotations

import textwrap

from passengersim.config import Config

from .choice_models import check_choice_model_restrictions
from .demand import check_demands_without_fares, check_demands_without_paths, check_reference_price_scaling
from .fares import check_fare_restrictions, check_fares_without_demands
from .markets import check_markets_without_fares, check_min_fare_price_by_market
from .todd import check_todd_curves


def _bad(s, e):
    print(f"\033[1;31m ❌ \033[0m{s}")
    print(textwrap.indent(str(e), "    "))


def _good(s):
    print(f"\033[1;32m ✅ \033[0m{s}")


def _warn(s):
    print(f"\033[1;33m 🚧 \033[0m{s}")


def check_config(cfg: Config):
    """Run various checks to ensure that the config is valid and reasonable."""

    try:
        check_fares_without_demands(cfg)
        _good("no fares without demands")
    except ValueError as e:
        _bad("fares without demands:", e)

    try:
        check_markets_without_fares(cfg)
        _good("no markets without fares")
    except ValueError as e:
        _bad("markets without fares:", e)

    try:
        check_demands_without_fares(cfg)
        _good("no demands without fares")
    except ValueError as e:
        _bad("demands without fares:", e)

    try:
        check_demands_without_paths(cfg)
        _good("no demands without paths")
    except ValueError as e:
        _bad("demands without paths:", e)

    r = check_fare_restrictions(cfg)
    if len(r):
        _good(f"fare restrictions: {len(r)} unique restrictions found")
        for k, v in r.items():
            print(f"   - {k}: {v} occurrences")
    else:
        _warn("no fare restrictions found")

    try:
        check_reference_price_scaling(cfg)
        _good("reference price scaling is consistent")
    except ValueError as e:
        _bad("reference price scaling:", e)

    r1 = check_choice_model_restrictions(cfg)
    if len(r1):
        _good(f"choice model restrictions: {len(r1)} unique restrictions found")
        for k, v in r1.items():
            print(f"   - {k}: {v}")
    else:
        _warn("no choice model restrictions found")

    if r.keys() != r1.keys():
        _bad(
            "choice model and fare restrictions don't match",
            f"{len(r)} fare restrictions vs {len(r1)} choice model restrictions",
        )
    else:
        _good("choice model and fare restrictions match")
