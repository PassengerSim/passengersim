from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Config


def preprocess_config(cfg: Config) -> Config:
    """Conduct cleaning and preprocessing on this Config.

    This will run the following steps:

    - connection builder
    - compute delta-t for all markets as needed
    - assign standard TODD curves to all demands if `simulation_controls.use_standard_todd_curves`
    """
    from passengersim.config.demands import assign_standard_todd_curves
    from passengersim.config.markets import compute_delta_t_for_markets
    from passengersim.config.todd_curves import clean_todd_curves
    from passengersim.connection_builder import prebuild_connections

    prebuild_connections(cfg)
    compute_delta_t_for_markets(cfg)
    assign_standard_todd_curves(cfg)
    clean_todd_curves(cfg)
    return cfg
