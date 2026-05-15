import warnings
from collections import defaultdict

from passengersim.config import Config


def check_fares_without_demands(cfg: Config, *, clean: bool = False, inplace: bool = True) -> Config:
    """Check for fares without demands from the config.

    Parameters
    ----------
    cfg : Config
    clean : bool, default False
        If True, remove fares without demands from the config. If False, raise an error if fares
        without fares are found.
     inplace : bool, default True
        If True and `clean` is also True, modify the input config in place. If False, return a
        modified copy of the config.  This has no effect if `clean` is False, since no modifications
        are made to the config in that case.

    Returns
    -------
    Config
         The input config, potentially modified to remove fares without demands if `clean` is True.

    Raises
    ------
    ValueError
        If fares without demands are found, and `clean` is False.
    """
    if not inplace:
        cfg = cfg.model_copy(deep=True)

    demands_by_market = defaultdict(list)
    for demand in cfg.demands:
        demands_by_market[f"{demand.orig}~{demand.dest}"].append(demand)

    fares_with_demand = []
    fares_without_demand = []
    for fare in cfg.fares:
        if f"{fare.orig}~{fare.dest}" in demands_by_market:
            fares_with_demand.append(fare)
        else:
            fares_without_demand.append(fare)

    if clean:
        if len(fares_without_demand):
            warnings.warn(f"Fares without demands: {len(fares_without_demand)}", stacklevel=2)
        cfg.fares = fares_with_demand
    else:
        if len(fares_without_demand):
            raise ValueError(
                f"found {len(fares_without_demand)} fares without demands, including {fares_without_demand[:3]}"
            )
    return cfg
