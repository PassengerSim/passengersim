from __future__ import annotations

from typing import Literal

from passengersim_core import Frat5, Path, PathClass


def clone_path(path: Path) -> Path:
    clone = Path(orig=path.orig, dest=path.dest, carrier=path.carrier)
    for pc in path.pathclasses:
        clone_pc = PathClass(pc.booking_class)
        clone_pc.fare = pc.fare
        clone_pc.history = pc.history
        clone_pc.forecast.history = pc.forecast.history
        clone.add_path_class(clone_pc)
    return clone


def conditional_q_forecast(
    path: Path,
    f5: Frat5,
    dcp_index: int = 0,
    regression_weight: Literal["sellup", "sellup^2", "none", None] = "sellup^2",
    max_cap: float = 10,
    partial_tf_weight_adjustment: bool = False,
) -> None:
    path.untruncate_demand(dcp_index, "em", which_data="yieldable")
    path.compute_simple_fare_adjustments(algorithm=None, frat5=f5)
    path.compute_conditional_q_forecast(
        frat5=f5,
        regression_weight=regression_weight,
        max_cap=max_cap,
        snapshot_instruction=None,
        partial_tf_weight_adjustment=partial_tf_weight_adjustment,
    )
    path.allocate_q_demand(frat5=f5, dcp_index=dcp_index)
    path.combine_forecasts()
