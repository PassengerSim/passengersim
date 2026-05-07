from __future__ import annotations

from typing import TYPE_CHECKING

from passengersim.config.paths import Path
from passengersim.driver import Simulation

if TYPE_CHECKING:
    from passengersim.config import Config


def prebuild_connections(cfg: Config, **kwargs) -> Config:

    sim = Simulation(cfg.model_copy(deep=len(kwargs) > 0))
    additional_kwds = {}
    for key, value in kwargs.items():
        if hasattr(sim.config.simulation_controls.connection_builder, key):
            setattr(sim.config.simulation_controls.connection_builder, key, value)
        else:
            additional_kwds[key] = value

    _num_paths = sim.eng.build_connections(**dict(sim.config.simulation_controls.connection_builder), **additional_kwds)

    path_defs = [
        Path.model_validate(
            {
                "path_id": p.path_id,
                "path_quality_index": p.path_quality_index,
                "orig": p.orig,
                "dest": p.dest,
                "legs": p.leg_ids,
            }
        )
        for p in sim.eng.paths
    ]

    cfg.paths.extend(path_defs)
    cfg.simulation_controls.connection_builder.existing_paths = "required"

    return cfg
