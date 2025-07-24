import pandas as pd

from passengersim.summary import SummaryTables

fc_target = {
    "AL1": {
        "Y0": 0.19113,
        "Y1": 0.18238,
        "Y2": 0.100455,
        "Y3": 0.0464455,
        "Y4": 0.360767,
        "Y5": 0.118815,
    },
    "AL2": {
        "Y0": 0.1912,
        "Y1": 0.18199,
        "Y2": 0.09998,
        "Y3": 0.04636,
        "Y4": 0.36074,
        "Y5": 0.11985,
    },
}

lf_target = {
    "AL1": {
        "0-49": 170,
        "50-54": 204,
        "55-59": 325,
        "60-64": 497,
        "65-69": 772,
        "70-74": 1037,
        "75-79": 1243,
        "80-84": 1585,
        "85-89": 1642,
        "90-94": 1842,
        "95-99": 2055,
        "100": 4628,
    },
    "AL2": {
        "0-49": 170,
        "50-54": 204,
        "55-59": 325,
        "60-64": 497,
        "65-69": 772,
        "70-74": 1037,
        "75-79": 1243,
        "80-84": 1585,
        "85-89": 1642,
        "90-94": 1842,
        "95-99": 2055,
        "100": 4628,
    },
}


class CalibrationScore:
    def __init__(
        self,
        target_fare_class_mix: pd.Series | dict[str, dict[str, float]],
        weight_fare_class_mix: pd.Series | dict[str, float],
        target_load_factor_distribution: pd.Series | dict[str, dict[str, float]],
        weight_load_factor_distribution: pd.Series | dict[str, float],
    ):
        """
        Score generator for simulation calibration.

        Parameters
        ----------
        target_fare_class_mix : pd.Series | dict
            Target fare class mix, by carrier and booking class.
            If a Series, must have a two level MultiIndex with
            "carrier" and "booking_class" levels.  If a dict,
            it should have nested dicts with carrier keys and
            then booking class keys.
        weight_fare_class_mix : pd.Series | dict
            Weights for fare class mix, by carrier.
        target_load_factor_distribution : pd.Series | dict
            Target load factor distribution, by carrier and load factor bin.
            If a Series, must have a two level MultiIndex with
            "carrier" and "load_factor_range" levels.  If a dict,
            it should have nested dicts with carrier keys and
            then load factor bin keys.
        weight_load_factor_distribution : pd.Series | dict
            Weights for load factor distribution, by carrier.
        """

        # targets for fare class mix, by carrier and booking class
        if isinstance(target_fare_class_mix, dict):
            self.target_fare_class_mix = (
                pd.DataFrame(target_fare_class_mix).rename_axis(columns="carrier", index="booking_class").unstack()
            )
        else:
            self.target_fare_class_mix = target_fare_class_mix
        assert self.target_fare_class_mix.index.names == ["carrier", "booking_class"]

        # targets for load factor distribution, by carrier and load factor bin
        if isinstance(target_load_factor_distribution, dict):
            self.target_load_factor_distribution = (
                pd.DataFrame(target_load_factor_distribution)
                .rename_axis(columns="carrier", index="load_factor_range")
                .unstack()
            )
        else:
            self.target_load_factor_distribution = target_load_factor_distribution
        assert self.target_load_factor_distribution.index.names == [
            "carrier",
            "load_factor_range",
        ]

        # ensure targets are normalized
        self.target_fare_class_mix = self.target_fare_class_mix.groupby("carrier").transform(lambda x: (x / x.sum()))
        self.target_load_factor_distribution = self.target_load_factor_distribution.groupby("carrier").transform(
            lambda x: (x / x.sum())
        )

        # Weights for fare class mix, by carrier
        if isinstance(weight_fare_class_mix, dict):
            self.wgt_fare_class_mix = (
                pd.DataFrame.from_dict(weight_fare_class_mix, orient="index", columns=["wgt"])
                .rename_axis(index="carrier")
                .iloc[:, 0]
            )
        else:
            self.wgt_fare_class_mix = weight_fare_class_mix

        # Weights for load factor distribution, by carrier
        if isinstance(weight_load_factor_distribution, dict):
            self.wgt_load_factor_distribution = (
                pd.DataFrame.from_dict(weight_load_factor_distribution, orient="index", columns=["wgt"])
                .rename_axis(index="carrier")
                .iloc[:, 0]
            )
        else:
            self.wgt_load_factor_distribution = weight_load_factor_distribution

    def _analyze_fare_class_mix(self, run_summary: SummaryTables) -> tuple[pd.Series, pd.Series, pd.Series]:
        run_shares = run_summary.raw_fare_class_mix["sold"].groupby("carrier").transform(lambda x: (x / x.sum()))
        tgt_shares = self.target_fare_class_mix
        diffs = run_shares - tgt_shares
        return run_shares, tgt_shares, diffs

    def compute_fare_class_mix_score(self, run_summary: SummaryTables) -> float:
        run_shares, tgt_shares, diffs = self._analyze_fare_class_mix(run_summary)
        return ((diffs * self.wgt_fare_class_mix) ** 2).sum()

    def analyze_fare_class_mix(self, run_summary: SummaryTables) -> pd.DataFrame:
        run_shares, tgt_shares, diffs = self._analyze_fare_class_mix(run_summary)
        return pd.concat(
            [
                run_shares.rename("simulation"),
                tgt_shares.rename("target"),
                diffs.rename("difference"),
            ],
            axis=1,
        )

    def _analyze_load_factor_distribution(self, run_summary: SummaryTables) -> tuple[pd.Series, pd.Series, pd.Series]:
        bins = sorted(
            set(
                int(k.split("-")[0])
                for k in self.target_load_factor_distribution.index.get_level_values("load_factor_range")
            )
        )
        run_lfs = (
            run_summary.fig_leg_load_factor_distribution(raw_df=True, breakpoints=bins)
            .rename(columns={"Load Factor Range": "load_factor_range"})
            .set_index(["carrier", "load_factor_range"])
            .iloc[:, 0]
        )
        # normalize load factor distribution
        run_lfs = run_lfs.groupby("carrier").transform(lambda x: (x / x.sum()))
        tgt_lfs = self.target_load_factor_distribution
        diffs = run_lfs - tgt_lfs
        return run_lfs, tgt_lfs, diffs

    def compute_load_factor_distribution_score(self, run_summary: SummaryTables) -> float:
        run_lfs, tgt_lfs, diffs = self._analyze_load_factor_distribution(run_summary)
        return ((diffs * self.wgt_load_factor_distribution) ** 2).sum()

    def analyze_load_factor_distribution(self, run_summary: SummaryTables) -> pd.DataFrame:
        run_lfs, tgt_lfs, diffs = self._analyze_load_factor_distribution(run_summary)
        return pd.concat(
            [
                run_lfs.rename("simulation"),
                tgt_lfs.rename("target"),
                diffs.rename("difference"),
            ],
            axis=1,
        )
