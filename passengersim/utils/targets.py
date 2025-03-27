from collections import defaultdict

import pandas as pd

from passengersim.config import Config
from passengersim.summary import SummaryTables


def target_demand_by_timeframe(cfg: Config):
    target_demand = defaultdict(lambda: defaultdict(lambda: 0))
    dcps = [None] + cfg.dcps
    for dcp0, dcp1 in zip(dcps, dcps[1:]):
        for dmd in cfg.demands:
            b = dmd.base_demand
            curve = cfg.booking_curves[dmd.curve].curve
            frac = curve.get(dcp1, 0) - curve.get(dcp0, 0)
            target_demand[dmd.segment][dcp1] += (
                frac * b * cfg.simulation_controls.demand_multiplier
            )
    return {i: dict(j) for (i, j) in target_demand.items()}


def computed_targets(cfg: Config):
    td = pd.DataFrame(target_demand_by_timeframe(cfg))
    td.loc[0, :] = 0
    td = td.cumsum().shift(1).fillna(0).add_prefix("avg_")
    td = td.assign(**{"carrier": "TGT", "booking_class": "XX"})
    td = td.rename_axis(index="days_prior").reset_index()
    return SummaryTables(
        bookings_by_timeframe=td,
    )
