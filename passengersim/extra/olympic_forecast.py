from typing import Literal

import numpy as np

from passengersim.rm_steps import RmStep


class OlympicForecastStep(RmStep):
    """
    In this step, we use a custom forecast algorithm, the "Olympic Average",
    implemented in Python.  In this forecast, the mean (mu) forecast for any
    bucket is computed by taking the 26 historical values, discarding the highest
    and lowest, and taking the mean of the remaining 24 values.

    This forecaster is meant primarily as a technology demonstration for how
    to implement a custom RM step in Python.
    """

    step_type: Literal["olympicforecast"]
    """
    Forecasting step type must be "olympicforecast".
    """

    algorithm: Literal["additive_pickup"] = "additive_pickup"
    """
    Forecasting algorithm, only `additive_pickup` is implemented.
    """

    kind: Literal["leg"] = "leg"
    """
    Level of collected demand data that should be used for forecasting.
    """

    @property
    def requires(self) -> list[str]:
        if self.kind == "leg":
            return ["leg_demand"]
        else:
            raise ValueError(f"bad kind {self.kind!r}")

    @property
    def produces(self) -> list[str]:
        if self.kind == "leg":
            return ["leg_forecast"]
        else:
            raise ValueError(f"bad kind {self.kind!r}")

    def serialize(self):
        return {
            "step_type": "forecast",
            "name": self.name,
            "algorithm": self.algorithm,
            "kind": self.kind,
        }

    def run(self, sim, carrier, dcp_index, dcp, debug=False):
        if sim.sample < 3:
            return
        if self.kind == "leg":
            for leg in sim.legs:
                if leg.carrier != carrier:
                    continue
                # days_prior = compute_days_prior(sim, leg.dep_time)
                for bkt in leg.buckets:
                    if self.algorithm == "additive_pickup":
                        self.additive_pickup(bkt, dcp_index)
                    else:
                        raise Exception(f"Unknown forecast algorithm: {self.algorithm}")
        else:
            raise Exception(f"Unknown forecast type: {self.kind}")

    def additive_pickup(self, bkt, dcp_index):
        # get historic demand
        history = get_bucket_history(bkt, dcp_index)
        # sum historic demand over all remaining time periods
        history = history.sum(1)
        # compute average pickup excluding min and max
        avg_pickup = (history.sum() - history.max() - history.min()) / (
            history.size - 2
        )
        # ensure non-negative value
        fcst = max(avg_pickup, 0.0)
        bkt.fcst_mean = fcst
        # compute standard dev as normal, including outliers
        bkt.fcst_std_dev = history.std()
        bkt.fcst_revenue = bkt.revenue / bkt.sold if bkt.sold > 0 else 0


def get_bucket_history(bkt, dcp_index):
    num_deps = bkt.get_history_num_dep()
    num_dcps = bkt.get_history_num_dcp()
    result = np.zeros([num_deps, num_dcps - dcp_index], dtype=np.float32)
    for dep in range(num_deps):
        j = 0
        for i in range(dcp_index, num_dcps):
            result[dep, j] = bkt.get_history_demand(dep, i)
            j += 1
    return result
