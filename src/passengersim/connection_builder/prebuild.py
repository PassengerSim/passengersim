from __future__ import annotations

from typing import TYPE_CHECKING

from passengersim.config.base import filterable_list
from passengersim.config.paths import Path
from passengersim.driver import Simulation

if TYPE_CHECKING:
    from passengersim.config import Config


def prebuild_connections(cfg: Config, *, inplace: bool = True, **kwargs) -> Config:
    """
    Prebuild connections on this config.

    Parameters
    ----------
    cfg : Config
        The config on which to build connections.
    inplace : bool
        Whether to modify the config in place or return a modified copy.  Default is True.
    max_legs : int, optional
        The maximum number of legs to include in any generated path, which
        should be between 1 and 6.
    max_legs_if_nonstop_exists : int
        The maximum number of legs to include in any generated path if a nonstop
        path exists in that market. The nonstop path can be on any carrier.
    existing_paths: {"keep", "replace", "required", "none"}
        What to do with existing paths when generating new ones.
        The default value is "keep", which means that for any market where paths
        already exist they will be used, and no new paths will be generated. For
        markets where no paths exist, new paths will be generated as normal.
        Alternatively, set this to "none", which has the same behavior as "keep"
        but will raise an error if the configuration includes any defined paths.
        Other options are "replace", which will remove all existing paths and then
        generating new ones for all markets, and "required", which will prevent
        the generation of any new paths.  If set to "required", the connection
        builder will only serve as a check that all markets have paths, and will
        raise an error if any market is missing paths.
    circuity_function : str
        The function to use when deciding if a path is allowable due to circuity.
        Circuity is the ratio of the total distance of the path to the direct distance
        between the origin and destination. The default function disallows paths that are
        excessively circuitous, with thresholds that vary based on the direct distance.
        Users can provide their own function with the same signature to implement custom
        circuity rules.
        The circuity function is specified by name here and should be a registered
        circuity function.  See `passengersim.connection_builder.circuity` for more
        details on circuity functions and how to register custom ones.
    nonstop_leg_path_id_alignment : bool = True
        Whether to align path IDs with leg IDs for nonstop paths.
        By default, this is set to True, which means that any nonstop path (corresponds
        to a single leg) will be assigned the same ID as that leg by the path building
        algorithm. This can make it easier to identify and analyze nonstop paths in the
        simulation results. If set to False, nonstop paths will be assigned unique IDs
        that do not necessarily align with leg IDs. This generally corresponds to the
        behavior of the previous path building algorithm, and may be desirable in cases
        where there are existing results to compare against.
    verbosity : int
        The level of detail to include in connection builder logging.
    min_paths_per_market : int
        The minimum number of paths to generate for each market.
        This is not a hard minimum, but the connection builder will make an effort to
        generate at least this many paths for each market, if possible given the other
        settings.  This could be by progressively relaxing circuity rules, maximum
        connection times, or other tweaks. If the connection builder is unable to generate
        at least this many paths for a market, it will log a warning.
    extra_max_connect_time_per_iteration : int
        Extra time added to all maximum connection times at each iteration.
        The connection builder iterates when the `min_paths_per_market` value is not
        met, potentially relaxing circuity rules at each iteration.  This setting also
        allows for the relaxation of maximum connect times, by adding this many minutes
        to all maximum connection times at each iteration.
    """
    if not inplace:
        cfg = cfg.model_copy(deep=True)
    sim_cfg = cfg.model_copy(deep=len(kwargs) > 0)
    additional_kwds = {}
    for key, value in kwargs.items():
        if hasattr(sim_cfg.simulation_controls.connection_builder, key):
            setattr(sim_cfg.simulation_controls.connection_builder, key, value)
        else:
            additional_kwds[key] = value
    sim = Simulation(sim_cfg)
    _num_paths = sim.eng.build_connections(
        **sim_cfg.simulation_controls.connection_builder.model_dump(), **additional_kwds
    )

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
    cfg.paths = filterable_list(cfg.paths)  # ensure paths are filterable

    return cfg
