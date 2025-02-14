from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from passengersim import Simulation


import pandas as pd

from .generic import GenericTracer
from .welford import Welford


class PathForecastTracer(GenericTracer):
    name: str = "selected_path_forecasts"
    path_forecasts: dict[int, Welford]

    def __init__(self, path_ids: list[int]):
        self.path_ids = path_ids
        self.reset()

    def reset(self) -> None:
        self.path_forecasts = {path_id: Welford() for path_id in self.path_ids}

    def fresh(self) -> PathForecastTracer:
        return PathForecastTracer(self.path_ids)

    def attach(self, sim: Simulation) -> None:
        sim.begin_sample_callback(self.fresh())

    def __call__(self, sim: Simulation):
        if sim.sim.sample < sim.sim.burn_samples:
            return
        for path_id in self.path_ids:
            fd = sim.sim.paths.select(path_id=path_id).get_forecast_data()
            self.path_forecasts[path_id].update(fd.to_dataframe())

    def finalize(self) -> pd.DataFrame:
        return pd.concat(
            {path_id: welford.mean for path_id, welford in self.path_forecasts.items()},
            names=["path_id"],
        )
