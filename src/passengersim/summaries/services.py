from __future__ import annotations

import pandas as pd

from passengersim.summaries.generic import GenericSimulationTables


class SimTabServices(GenericSimulationTables):
    @property
    def services(self) -> pd.DataFrame:
        """Service-level summary data, aggregated by carrier and operating leg o-d.

        A 'service' is an aggregation of all legs sharing a unique combination of
        carrier, origin, and destination.  This table aggregates data from the legs
        table to the service level.
        """
        if "services" not in self._data:
            df = self.legs.groupby(["carrier", "orig", "dest"]).agg(
                frequency=("gt_sold", "count"),
                distance=("distance", "mean"),
                gt_sold=("gt_sold", "sum"),
                gt_capacity=("gt_capacity", "sum"),
                gt_sold_local=("gt_sold_local", "sum"),
                gt_revenue=("gt_revenue", "sum"),
            )
            df["capacity"] = df["gt_capacity"] / self.n_total_samples
            df["avg_sold"] = df["gt_sold"] / self.n_total_samples
            df["avg_sold_local"] = df["gt_sold_local"] / self.n_total_samples
            df["avg_revenue"] = df["gt_revenue"] / self.n_total_samples
            df["avg_load_factor"] = 100.0 * df["gt_sold"] / df["gt_capacity"]
            df["avg_local"] = 100.0 * df["gt_sold_local"] / df["gt_sold"]
            self._data["services"] = df
        return self._data["services"]
