from __future__ import annotations

import ast
import logging
import os.path
import pathlib
from collections.abc import Collection
from typing import Literal

import numpy as np
import pandas as pd

from . import database
from .reporting import report_figure

logger = logging.getLogger("passengersim.summary")


class SummaryTables:
    @classmethod
    def from_sqlite(
        cls,
        filename: str | pathlib.Path,
        make_indexes: bool | dict = False,
        additional: Collection[str | tuple] | str | None = None,
    ):
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        db = database.Database(
            engine="sqlite",
            filename=filename,
        )

        demands = cls.load_basic_table(db, "demand_summary")
        legs = cls.load_basic_table(db, "leg_summary")
        paths = cls.load_basic_table(db, "path_summary")
        carriers = cls.load_basic_table(db, "carrier_summary")

        summary = cls(
            demands=demands,
            legs=legs,
            paths=paths,
            carriers=carriers,
        )

        if make_indexes:
            if isinstance(make_indexes, dict):
                db.add_indexes(**make_indexes)
            else:
                db.add_indexes()

        logger.info("loading configs")
        config = db.load_configs()

        summary.load_additional_tables(
            db,
            scenario=config.scenario,
            burn_samples=config.simulation_controls.burn_samples,
            additional=additional,
        )
        summary.cnx = db
        return summary

    @classmethod
    def aggregate(cls, summaries: Collection[SummaryTables]):
        """Aggregate multiple summary tables."""
        if not summaries:
            return None

        # dataframes where trial is in the index, just concatenate
        def concat(name):
            frames = []
            for s in summaries:
                frame = getattr(s, name)
                if frame is not None:
                    frames.append(frame)
            if frames:
                return pd.concat(frames)
            return None

        carrier_history = concat("carrier_history")
        bookings_by_timeframe = concat("bookings_by_timeframe")
        demand_to_come = concat("demand_to_come")

        # demands has some columns that are averages and some that are sums
        demands_avg = sum(
            s.demands.set_index(["orig", "dest", "segment"])[
                ["sold", "revenue", "avg_fare"]
            ]
            for s in summaries
        ) / len(summaries)
        demands_sum = sum(
            s.demands.set_index(["orig", "dest", "segment"])[
                ["gt_demand", "gt_sold", "gt_revenue"]
            ]
            for s in summaries
        )
        demands = pd.concat([demands_avg, demands_sum], axis=1).reset_index()

        carriers = sum(s.carriers.set_index("carrier") for s in summaries) / len(
            summaries
        )
        legs = sum(
            s.legs.set_index(["carrier", "flt_no", "orig", "dest"]) for s in summaries
        ) / len(summaries)
        paths = sum(
            s.paths.set_index(["orig", "dest", "carrier1", "flt_no1", "carrier2"])
            for s in summaries
        ) / len(summaries)

        def average(name):
            frames = []
            for s in summaries:
                frame = getattr(s, name)
                if frame is not None:
                    frames.append(frame)
            if frames:
                return sum(frames) / len(frames)
            return None

        fare_class_mix = average("fare_class_mix")
        leg_forecasts = average("leg_forecasts")
        path_forecasts = average("path_forecasts")
        bid_price_history = average("bid_price_history")
        displacement_history = average("displacement_history")

        result = cls(
            demands=demands,
            legs=legs,
            paths=paths,
            carriers=carriers,
            fare_class_mix=fare_class_mix,
            leg_forecasts=leg_forecasts,
            path_forecasts=path_forecasts,
            carrier_history=carrier_history,
            bookings_by_timeframe=bookings_by_timeframe,
            bid_price_history=bid_price_history,
            displacement_history=displacement_history,
            demand_to_come=demand_to_come,
        )
        result.meta_trials = summaries
        return result

    @classmethod
    def load_basic_table(self, db: database.Database, tablename: str):
        """Load a basic table"""
        logger.info("loading %s", tablename)
        return db.dataframe(f"SELECT * FROM {tablename}")

    def load_additional_tables(
        self,
        db: database.Database,
        scenario: str,
        burn_samples: int,
        additional: Collection[str | tuple] | str | None = (
            "fare_class_mix",
            "bookings_by_timeframe",
            "total_demand",
            "load_factor_distribution",
        ),
    ) -> None:
        """
        Load additional summary tables based on common queries.

        Parameters
        ----------
        db : Database
        scenario : str
        burn_samples : int
            The number of samples in the burn period.  The data from these samples
            is ignored in most common queries.
        additional : Collection[str | tuple] | str
            One or more additional tables to load.  If "*", then this will load
            all common queries supported by the configuration used during the
            simulation.
        """
        if isinstance(additional, str):
            if additional == "*":
                additional = set()
                cfg = db.load_configs(scenario)
                if "fare" in cfg.db.write_items:
                    additional.add("fare_class_mix")
                if "fare_final" in cfg.db.write_items:
                    additional.add("fare_class_mix")
                if "bookings" in cfg.db.write_items:
                    additional.add("bookings_by_timeframe")
                if "demand" in cfg.db.write_items:
                    additional.add("total_demand")
                    additional.add("demand_to_come")
                if "demand_final" in cfg.db.write_items:
                    additional.add("total_demand")
                if "bucket" in cfg.db.write_items:
                    additional.add("leg_forecasts")
                    additional.add("carrier_history")
                if "pathclass" in cfg.db.write_items:
                    additional.add("path_forecasts")
                    additional.add("local_and_flow_yields")
                if "pathclass_final" in cfg.db.write_items:
                    additional.add("local_and_flow_yields")
                if "leg" in cfg.db.write_items and cfg.db.store_leg_bid_prices:
                    additional.add("bid_price_history")
                if "leg" in cfg.db.write_items and cfg.db.store_displacements:
                    additional.add("displacement_history")
                if "leg" in cfg.db.write_items or "leg_final" in cfg.db.write_items:
                    additional.add("load_factor_distribution")
            else:
                additional = [additional]
        elif additional is None:
            additional = []

        if "fare_class_mix" in additional and db.is_open:
            logger.info("loading fare_class_mix")
            self.fare_class_mix = database.common_queries.fare_class_mix(
                db, scenario, burn_samples=burn_samples
            )
            if self.od_fare_class_mix:
                for orig, dest in list(self.od_fare_class_mix):
                    self.od_fare_class_mix[(orig, dest)] = (
                        database.common_queries.od_fare_class_mix(
                            db, orig, dest, scenario, burn_samples=burn_samples
                        )
                    )
        # load additional fare class mix tables
        for i in additional:
            if isinstance(i, tuple) and i[0] == "od_fare_class_mix" and db.is_open:
                orig, dest = i[1], i[2]
                if self.od_fare_class_mix is None:
                    self.od_fare_class_mix = {}
                logger.info(f"loading od_fare_class_mix({orig},{dest})")
                self.od_fare_class_mix[(orig, dest)] = (
                    database.common_queries.od_fare_class_mix(
                        db, orig, dest, scenario, burn_samples=burn_samples
                    )
                )

        for i in additional:
            cutoffs = None
            if i == "load_factor_distribution" and db.is_open:
                cutoffs = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)  # default cutoffs
            elif (
                isinstance(i, tuple)
                and i[0] == "load_factor_distribution"
                and db.is_open
            ):
                cutoffs = ast.literal_eval(i[1])
            if cutoffs is not None:
                logger.info("loading load_factor_distribution")
                self.load_factor_distribution = (
                    database.common_queries.load_factor_distribution(
                        db, scenario, burn_samples=burn_samples, cutoffs=cutoffs
                    )
                )

        if "bookings_by_timeframe" in additional and db.is_open:
            logger.info("loading bookings_by_timeframe")
            self.bookings_by_timeframe = database.common_queries.bookings_by_timeframe(
                db, scenario, burn_samples=burn_samples
            )

        if "total_demand" in additional and db.is_open:
            logger.info("loading total_demand")
            self.total_demand = database.common_queries.total_demand(
                db, scenario, burn_samples
            )

        if "leg_forecasts" in additional and db.is_open:
            logger.info("loading leg_forecasts")
            self.leg_forecasts = database.common_queries.leg_forecasts(
                db, scenario, burn_samples
            )

        if "path_forecasts" in additional and db.is_open:
            logger.info("loading path_forecasts")
            self.path_forecasts = database.common_queries.path_forecasts(
                db, scenario, burn_samples
            )

        if "demand_to_come" in additional and db.is_open:
            logger.info("loading demand_to_come")
            self.demand_to_come = database.common_queries.demand_to_come(db, scenario)

        if "carrier_history" in additional and db.is_open:
            logger.info("loading carrier_history")
            self.carrier_history = database.common_queries.carrier_history(db, scenario)

        if "bid_price_history" in additional and db.is_open:
            logger.info("loading bid_price_history")
            self.bid_price_history = database.common_queries.bid_price_history(
                db, scenario, burn_samples
            )

        if "displacement_history" in additional and db.is_open:
            logger.info("loading displacement_history")
            self.displacement_history = database.common_queries.displacement_history(
                db, scenario, burn_samples
            )

        if "local_and_flow_yields" in additional and db.is_open:
            logger.info("loading local_and_flow_yields")
            self.local_and_flow_yields = database.common_queries.local_and_flow_yields(
                db, scenario, burn_samples
            )

        if "leg_local_and_flow_by_class" in additional and db.is_open:
            logger.info("loading leg_local_and_flow_by_class")
            self.leg_carried = database.common_queries.leg_local_and_flow_by_class(
                db, scenario, burn_samples
            )

    def __init__(
        self,
        name: str | None = "name?",
        demands: pd.DataFrame | None = None,
        fares: pd.DataFrame | None = None,
        legs: pd.DataFrame | None = None,
        paths: pd.DataFrame | None = None,
        path_classes: pd.DataFrame | None = None,
        carriers: pd.DataFrame | None = None,
        fare_class_mix: pd.DataFrame | None = None,
        load_factors: pd.DataFrame | None = None,
        bookings_by_timeframe: pd.DataFrame | None = None,
        total_demand: float | None = None,
        od_fare_class_mix: dict[tuple[str, str], pd.DataFrame] | None = None,
        leg_forecasts: pd.DataFrame | None = None,
        path_forecasts: pd.DataFrame | None = None,
        carrier_history: pd.DataFrame | None = None,
        demand_to_come: pd.DataFrame | None = None,
        bid_price_history: pd.DataFrame | None = None,
        displacement_history: pd.DataFrame | None = None,
        local_and_flow_yields: pd.DataFrame | None = None,
        leg_carried: pd.DataFrame | None = None,
        load_factor_distribution: pd.DataFrame | None = None,
    ):
        self.demands = demands
        self.fares = fares
        self.legs = legs
        self.paths = paths
        self.path_classes = path_classes
        self.carriers = carriers
        self.fare_class_mix = fare_class_mix
        self.od_fare_class_mix = od_fare_class_mix
        self.load_factors = load_factors
        self.bookings_by_timeframe = bookings_by_timeframe
        self.total_demand = total_demand
        self.leg_forecasts = leg_forecasts
        self.path_forecasts = path_forecasts
        self.carrier_history = carrier_history
        self.demand_to_come = demand_to_come
        self.bid_price_history = bid_price_history
        self.displacement_history = displacement_history
        self.local_and_flow_yields = local_and_flow_yields
        self.leg_carried = leg_carried
        self.load_factor_distribution = load_factor_distribution

    def to_records(self) -> dict[str, list[dict]]:
        """Convert all summary tables to a dictionary of records."""
        return {k: v.to_dict(orient="records") for (k, v) in self.__dict__.items()}

    def to_xlsx(self, filename: str | pathlib.Path) -> None:
        """Write summary tables to excel.

        Parameters
        ----------
        filename : Path-like
            The excel file to write.
        """
        if isinstance(filename, str):
            filename = pathlib.Path(filename)
        filename.parent.mkdir(exist_ok=True, parents=True)
        with pd.ExcelWriter(filename) as writer:
            for k, v in self.__dict__.items():
                if isinstance(v, pd.DataFrame):
                    v.to_excel(writer, sheet_name=k)

    def to_dataframe(self, table) -> pd.DataFrame:
        """Convert the summary tables to a individual dataframes."""
        sheet_count = 0
        for k, v in self.__dict__.items():
            if isinstance(v, pd.DataFrame):
                sheet_count += 1
                if sheet_count == table:
                    return v.assign(table=k)

        raise IndexError("There are fewer than", table, " DataFrames in the object")

    def aggregate_demand_history(self, by_segment: bool = True) -> pd.Series:
        """
        Total demand by sample, aggregated over all markets.

        Parameters
        ----------
        by_segment : bool, default True
            Aggregate by segment.  If false, segments are also aggregated.

        Returns
        -------
        pandas.Series
            Total demand, indexed by trial, sample, and segment
            (business/leisure).
        """
        groupbys = ["trial", "sample"]
        if by_segment:
            groupbys.append("segment")
        return self.demand_to_come.iloc[:, 0].groupby(groupbys, observed=False).sum()

    def demand_in_tf(self) -> pd.DataFrame | None:
        """History of demand arriving in each timeframe.

        This dataframe is derived from the `demand_to_come` dataframe
        by taking the sequential differences.
        """
        if self.demand_to_come is None:
            return None
        return self.demand_to_come.diff(-1, axis=1).iloc[:, :-1]

    def fig_carrier_mileage(self, raw_df: bool = False, report=None):
        """
        Figure showing ASM, RPM by carrier.

        ASM is available seat miles.  RPM is revenue passenger miles.

        Parameters
        ----------
        raw_df : bool, default False
            Return the raw data for this figure as a pandas DataFrame, instead
            of generating the figure itself.
        report : xmle.Reporter, optional
            Also append this figure to the given report.
        """
        df = (
            self.carriers.reset_index()[["carrier", "asm", "rpm"]]
            .set_index("carrier")
            .rename_axis(columns="measure")
            .unstack()
            .to_frame("value")
            .reset_index()
        )
        if raw_df:
            return df
        import altair as alt

        chart = alt.Chart(df, title="Carrier Loads")
        bars = chart.mark_bar().encode(
            x=alt.X("carrier:N", title="Carrier"),
            y=alt.Y("value", stack=None, title="miles"),
            color="measure",
            tooltip=["carrier", "measure", alt.Tooltip("value", format=".4s")],
        )
        text = chart.mark_text(
            dx=0,
            dy=5,
            color="white",
            baseline="top",
        ).encode(
            x=alt.X("carrier:N"),
            y=alt.Y("value").stack(None),
            text=alt.Text("value:Q", format=".4s"),
        )
        fig = (
            (bars + text)
            .properties(
                width=400,
                height=300,
            )
            .configure_axis(
                labelFontSize=12,
                titleFontSize=12,
            )
            .configure_legend(
                titleFontSize=12,
                labelFontSize=15,
            )
        )
        if report:
            report.add_figure(fig)
        return fig

    def _fig_fare_class_mix(
        self, df: pd.DataFrame, label_threshold: float = 0.06, title=None
    ):
        import altair as alt

        label_threshold_value = (
            df.groupby("carrier").avg_sold.sum().max() * label_threshold
        )
        chart = alt.Chart(
            df, **({"title": title} if title else {})
        ).transform_calculate(
            halfsold="datum.avg_sold / 2.0",
        )
        bars = chart.mark_bar().encode(
            x=alt.X("carrier:N", title="Carrier"),
            y=alt.Y("avg_sold:Q", title="Seats").stack("zero"),
            color="booking_class",
            tooltip=[
                "carrier",
                "booking_class",
                alt.Tooltip("avg_sold", format=".2f"),
            ],
        )
        text = chart.mark_text(dx=0, dy=3, color="white", baseline="top").encode(
            x=alt.X("carrier:N", title="Carrier"),
            y=alt.Y("avg_sold:Q", title="Seats").stack("zero"),
            text=alt.Text("avg_sold:Q", format=".2f"),
            opacity=alt.condition(
                f"datum.avg_sold < {label_threshold_value:.3f}",
                alt.value(0),
                alt.value(1),
            ),
            order=alt.Order("booking_class:N", sort="descending"),
        )
        return (
            (bars + text)
            .properties(
                width=400,
                height=300,
            )
            .configure_axis(
                labelFontSize=12,
                titleFontSize=12,
            )
            .configure_legend(
                titleFontSize=12,
                labelFontSize=15,
            )
        )

    @report_figure
    def fig_fare_class_mix(self, raw_df=False, label_threshold=0.06):
        df = self.fare_class_mix.reset_index()[["carrier", "booking_class", "avg_sold"]]
        if raw_df:
            return df
        return self._fig_fare_class_mix(
            df,
            label_threshold=label_threshold,
            title="Fare Class Mix",
        )

    @report_figure
    def fig_od_fare_class_mix(
        self, orig: str, dest: str, raw_df=False, label_threshold=0.06
    ):
        df = self.od_fare_class_mix[orig, dest].reset_index()[
            ["carrier", "booking_class", "avg_sold"]
        ]
        if raw_df:
            return df
        return self._fig_fare_class_mix(
            df, label_threshold=label_threshold, title=f"Fare Class Mix ({orig}-{dest})"
        )

    @report_figure
    def fig_load_factor_distribution(self, by_carrier: bool | str = True, raw_df=False):
        if not hasattr(self, "load_factor_distribution"):
            raise AttributeError(
                "load_factor_distribution data not found. Please load it first."
            )

        df_for_chart = self.load_factor_distribution
        df_for_chart.columns.names = ["Load Factor Range"]
        df_for_chart = df_for_chart.set_index("carrier")
        df_for_chart = df_for_chart.stack().rename("Count").reset_index()

        if not by_carrier:
            df_for_chart = (
                df_for_chart.groupby(["Load Factor Range"]).Count.sum().reset_index()
            )
        elif isinstance(by_carrier, str):
            df_for_chart = df_for_chart[df_for_chart["carrier"] == by_carrier]
            df_for_chart = df_for_chart.drop(columns=["carrier"])

        if raw_df:
            return df_for_chart

        import altair as alt

        if by_carrier is True:
            chart = (
                alt.Chart(df_for_chart)
                .mark_bar()
                .encode(
                    x=alt.X("Load Factor Range", title="Load Factor Range"),
                    y=alt.Y("Count:Q", title="Count"),
                    facet=alt.Facet("carrier:N", columns=2, title="Carrier"),
                    tooltip=[
                        alt.Tooltip("carrier", title="Carrier"),
                        alt.Tooltip("Count", title="Count"),
                    ],
                )
                .properties(
                    width=300, height=250, title="Leg Load Factor Frequency by Carrier"
                )
            )
        else:
            chart = (
                alt.Chart(df_for_chart)
                .mark_bar()
                .encode(
                    x=alt.X("Load Factor Range", title="Load Factor Range"),
                    y=alt.Y("Count:Q", title="Count"),
                )
                .properties(
                    width=600,
                    height=400,
                    title="Leg Load Factor Frequency"
                    if not by_carrier
                    else f"Leg Load Factor Frequency ({by_carrier})",
                )
            )

        return chart

    @property
    def raw_fare_class_mix(self) -> pd.DataFrame:
        """Raw data giving the fare class mix.

        This tidy dataframe contains these columns:

        - carrier (str)
        - booking_class (str)
        - avg_sold (float)
        """
        return self.fig_fare_class_mix(raw_df=True)

    @report_figure
    def fig_bookings_by_timeframe(
        self,
        by_carrier: bool | str = True,
        by_class: bool | str = False,
        raw_df: bool = False,
        errorbands: bool = False,
        exclude_nogo: bool = True,
    ):
        if errorbands:
            if by_carrier is True:
                raise NotImplementedError("error bands for all airlines is messy")
            return self._fig_bookings_by_timeframe_errorband(
                by_carrier=by_carrier, raw_df=raw_df
            )

        def differs(x):
            return x.shift(-1, fill_value=0) - x

        def _summarize(x: pd.DataFrame, c: str):
            if "trial" not in x.columns:
                x = x.assign(trial=0)
            if by_class:
                y = (
                    x.groupby(["trial", "carrier", "booking_class", "days_prior"])[
                        f"avg_{c}"
                    ]
                    .sum()
                    .unstack(["trial", "carrier", "booking_class"])
                    .sort_index(ascending=False)
                    .apply(differs)
                    .stack(["carrier", "booking_class"])
                    .aggregate(["mean", "sem"], axis=1)
                    .assign(
                        ci0=lambda x: np.maximum(x["mean"] - 1.96 * x["sem"], 0),
                        ci1=lambda x: x["mean"] + 1.96 * x["sem"],
                    )
                )
            else:
                y = (
                    x.groupby(["trial", "carrier", "days_prior"])[f"avg_{c}"]
                    .sum()
                    .unstack(["trial", "carrier"])
                    .sort_index(ascending=False)
                    .apply(differs)
                    .stack("carrier")
                    .aggregate(["mean", "sem"], axis=1)
                    .assign(
                        ci0=lambda x: np.maximum(x["mean"] - 1.96 * x["sem"], 0),
                        ci1=lambda x: x["mean"] + 1.96 * x["sem"],
                    )
                )
            return pd.concat({c: y}, names=["paxtype"])

        bookings_by_timeframe = self.bookings_by_timeframe.reset_index()
        df0 = _summarize(bookings_by_timeframe, "business")
        df1 = _summarize(bookings_by_timeframe, "leisure")
        df = (
            pd.concat([df0, df1], axis=0)
            .rename(columns={"mean": "sold"})
            .reset_index()
            .query("(days_prior>0) & (sold>0)")
        )
        title = "Bookings by Timeframe"
        if by_class is True:
            title = "Bookings by Timeframe and Booking Class"
        title_annot = []
        if not by_carrier:
            g = ["days_prior", "paxtype"]
            if by_class:
                g += ["booking_class"]
            df = df.groupby(g)[["sold", "ci0", "ci1"]].sum().reset_index()
        if isinstance(by_carrier, str):
            df = df[df["carrier"] == by_carrier]
            df = df.drop(columns=["carrier"])
            title_annot.append(by_carrier)
            by_carrier = False
        if isinstance(by_class, str):
            df = df[df["booking_class"] == by_class]
            df = df.drop(columns=["booking_class"])
            title_annot.append(f"Class {by_class}")
            by_class = False
        if title_annot:
            title = f"{title} ({', '.join(title_annot)})"
        if exclude_nogo and "carrier" in df.columns:
            df = df[df["carrier"] != "NONE"]
        if raw_df:
            return df

        import altair as alt

        if by_carrier:
            color = "carrier:N"
            color_title = "Carrier"
        elif by_class:
            color = "booking_class:N"
            color_title = "Booking Class"
        else:
            color = "paxtype:N"
            color_title = "Passenger Type"

        if by_class:
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    color=alt.Color(color).title(color_title),
                    x=alt.X("days_prior:O")
                    .scale(reverse=True)
                    .title("Days Prior to Departure"),
                    y=alt.Y("sold"),
                    tooltip=(
                        [alt.Tooltip("carrier").title("Carrier")] if by_carrier else []
                    )
                    + [
                        alt.Tooltip("paxtype", title="Passenger Type"),
                        alt.Tooltip("days_prior", title="DfD"),
                        alt.Tooltip("sold", format=".2f"),
                    ],
                )
                .properties(
                    width=500,
                    height=200,
                )
                .facet(
                    row=alt.Row("paxtype:N", title="Passenger Type"),
                    title=title,
                )
                .configure_title(fontSize=18)
            )
        else:
            chart = (
                alt.Chart(df, title=title)
                .mark_line()
                .encode(
                    color=alt.Color(color).title(color_title),
                    x=alt.X("days_prior:O")
                    .scale(reverse=True)
                    .title("Days Prior to Departure"),
                    y=alt.Y("sold") if by_class else "sold",
                    strokeDash=alt.StrokeDash("paxtype").title("Passenger Type"),
                    tooltip=(
                        [alt.Tooltip("carrier").title("Carrier")] if by_carrier else []
                    )
                    + [
                        alt.Tooltip("paxtype", title="Passenger Type"),
                        alt.Tooltip("days_prior", title="DfD"),
                        alt.Tooltip("sold", format=".2f"),
                    ],
                )
                .properties(
                    width=500,
                    height=300,
                )
                .configure_axis(
                    labelFontSize=12,
                    titleFontSize=12,
                )
                .configure_legend(
                    titleFontSize=12,
                    labelFontSize=15,
                )
            )
        return chart

    def _fig_bookings_by_timeframe_errorband(
        self, by_carrier: bool | str = True, raw_df=False
    ):
        def differs(x):
            return x.shift(-1, fill_value=0) - x

        b = self.bookings_by_timeframe.reset_index()

        def _summarize(x, c):
            y = (
                x.groupby(["trial", "carrier", "days_prior"])[f"avg_{c}"]
                .sum()
                .unstack(["trial", "carrier"])
                .sort_index(ascending=False)
                .apply(differs)
                .stack("carrier")
                .aggregate(["mean", "sem"], axis=1)
                .assign(
                    ci0=lambda x: x["mean"] - 1.96 * x["sem"],
                    ci1=lambda x: x["mean"] + 1.96 * x["sem"],
                )
            )
            return pd.concat({c: y}, names=["paxtype"])

        df0 = _summarize(b, "business")
        df1 = _summarize(b, "leisure")
        df = (
            pd.concat([df0, df1], axis=0)
            .rename(columns={"mean": "sold"})
            .reset_index()
            .query("days_prior>0")
        )
        if not by_carrier:
            df = (
                df.groupby(["days_prior", "paxtype"])[["sold", "ci0", "ci1"]]
                .sum()
                .reset_index()
            )
        if isinstance(by_carrier, str):
            df = df[df["carrier"] == by_carrier]
            df = df.drop(columns=["carrier"])
            by_carrier = False
        if raw_df:
            return df
        import altair as alt

        chart = alt.Chart(df)
        lines = chart.mark_line().encode(
            color=alt.Color("carrier:N" if by_carrier else "paxtype").title(
                "Carrier" if by_carrier else "Passenger Type"
            ),
            x=alt.X("days_prior:O")
            .scale(reverse=True)
            .title("Days Prior to Departure"),
            y="sold",
            strokeDash=alt.StrokeDash("paxtype").title("Passenger Type"),
            tooltip=([alt.Tooltip("carrier").title("Carrier")] if by_carrier else [])
            + [
                alt.Tooltip("paxtype", title="Passenger Type"),
                alt.Tooltip("days_prior", title="DfD"),
                alt.Tooltip("sold", format=".2f"),
            ],
        )
        bands = chart.mark_errorband().encode(
            color=alt.Color(
                "carrier:N" if by_carrier else "paxtype",
                title="Carrier" if by_carrier else "Passenger Type",
            ),
            x=alt.X("days_prior:O")
            .scale(reverse=True)
            .title("Days Prior to Departure"),
            y="ci0",
            y2="ci1",
            strokeDash=alt.StrokeDash("paxtype").title("Passenger Type"),
        )

        return (
            (lines + bands)
            .properties(
                width=500,
                height=300,
            )
            .configure_axis(
                labelFontSize=12,
                titleFontSize=12,
            )
            .configure_legend(
                titleFontSize=12,
                labelFontSize=15,
            )
        )

    def _fig_carrier_load_factors(
        self,
        raw_df: bool,
        load_measure: str,
        measure_name: str,
        measure_format: str = ".2f",
        orient: Literal["h", "v"] = "h",
        title: str | None = None,
    ):
        df = self.carriers.reset_index()[["carrier", load_measure]]
        if raw_df:
            return df
        import altair as alt

        chart = alt.Chart(df)
        if orient == "v":
            bars = chart.mark_bar().encode(
                x=alt.X("carrier:N", title="Carrier"),
                y=alt.Y(f"{load_measure}:Q", title=measure_name).stack("zero"),
                tooltip=[
                    alt.Tooltip("carrier", title="Carrier"),
                    alt.Tooltip(
                        f"{load_measure}:Q", title=measure_name, format=measure_format
                    ),
                ],
            )
            text = chart.mark_text(dx=0, dy=3, color="white", baseline="top").encode(
                x=alt.X("carrier:N", title="Carrier"),
                y=alt.Y(f"{load_measure}:Q", title=measure_name).stack("zero"),
                text=alt.Text(f"{load_measure}:Q", format=measure_format),
            )
        else:
            bars = chart.mark_bar().encode(
                y=alt.Y("carrier:N", title="Carrier"),
                x=alt.X(f"{load_measure}:Q", title=measure_name).stack("zero"),
                tooltip=[
                    alt.Tooltip("carrier", title="Carrier"),
                    alt.Tooltip(
                        f"{load_measure}:Q", title=measure_name, format=measure_format
                    ),
                ],
            )
            text = chart.mark_text(
                dx=-5, dy=0, color="white", baseline="middle", align="right"
            ).encode(
                y=alt.Y("carrier:N", title="Carrier"),
                x=alt.X(f"{load_measure}:Q", title=measure_name).stack("zero"),
                text=alt.Text(f"{load_measure}:Q", format=measure_format),
            )
        fig = (
            (bars + text)
            .properties(
                width=500,
                height=10 + 20 * len(df),
            )
            .configure_axis(
                labelFontSize=12,
                titleFontSize=12,
            )
            .configure_legend(
                titleFontSize=12,
                labelFontSize=15,
            )
        )
        if title:
            fig.title = title
        return fig

    @report_figure
    def fig_carrier_load_factors(
        self, raw_df=False, load_measure: Literal["sys_lf", "avg_leg_lf"] = "sys_lf"
    ):
        measure_name = (
            "System Load Factor" if load_measure == "sys_lf" else "Leg Load Factor"
        )
        return self._fig_carrier_load_factors(
            raw_df,
            load_measure,
            measure_name,
            title=f"Carrier {measure_name}s",
        )

    @report_figure
    def fig_carrier_revenues(self, raw_df=False):
        return self._fig_carrier_load_factors(
            raw_df, "avg_rev", "Average Revenue", "$.4s", title="Carrier Revenues"
        )

    @report_figure
    def fig_carrier_yields(self, raw_df=False):
        return self._fig_carrier_load_factors(
            raw_df, "yield", "Average Yield", "$.4f", title="Carrier Yields"
        )

    @report_figure
    def fig_carrier_total_bookings(self, raw_df=False):
        return self._fig_carrier_load_factors(
            raw_df, "sold", "Total Bookings", ".4s", title="Carrier Total Bookings"
        )

    def _fig_forecasts(
        self,
        df,
        facet_on=None,
        y="forecast_mean",
        color="booking_class:N",
        y_title="Avg Demand Forecast",
    ):
        import altair as alt

        encoding = dict(
            x=alt.X("days_prior:O")
            .scale(reverse=True)
            .title("Days Prior to Departure"),
            y=alt.Y(f"{y}:Q", title=y_title),
        )
        if color:
            encoding["color"] = color
        if not facet_on:
            return alt.Chart(df).mark_line().encode(**encoding)
        else:
            return (
                alt.Chart(df)
                .mark_line()
                .encode(**encoding)
                .facet(
                    facet=f"{facet_on}:N",
                    columns=3,
                )
            )

    @report_figure
    def fig_leg_forecasts(
        self,
        by_flt_no: bool | int = True,
        by_class: bool | str = True,
        of: Literal["mu", "sigma"] | list[Literal["mu", "sigma"]] = "mu",
        raw_df=False,
    ):
        if isinstance(of, list):
            if raw_df:
                raise NotImplementedError
            fig = self.fig_leg_forecasts(
                by_flt_no=by_flt_no,
                by_class=by_class,
                of=of[0],
            )
            for of_ in of[1:]:
                fig |= self.fig_leg_forecasts(
                    by_flt_no=by_flt_no,
                    by_class=by_class,
                    of=of_,
                )
            return fig
        y = "forecast_mean" if of == "mu" else "forecast_stdev"
        columns = [
            "carrier",
            "flt_no",
            "booking_class",
            "days_prior",
            y,
        ]
        if self.leg_forecasts is None:
            raise ValueError("the leg_forecasts summary table is not available")
        df = self.leg_forecasts.reset_index()[columns]
        color = "booking_class:N"
        if isinstance(by_flt_no, int) and by_flt_no is not True:
            df = df[df.flt_no == by_flt_no]
        if isinstance(by_class, str):
            df = df[df.booking_class == by_class]
            color = None
        if raw_df:
            return df
        return self._fig_forecasts(
            df,
            facet_on=None,
            y=y,
            color=color,
            y_title="Mean Demand Forecast" if of == "mu" else "Std Dev Demand Forecast",
        )

    @report_figure
    def fig_path_forecasts(
        self,
        by_path_id: bool | int = True,
        by_class: bool | str = True,
        of: Literal["mu", "sigma", "closed"] = "mu",
        raw_df=False,
    ):
        if self.path_forecasts is None:
            raise ValueError("the path_forecasts summary table is not available")
        of_columns = {
            "mu": "forecast_mean",
            "sigma": "forecast_stdev",
            "closed": "forecast_closed_in_tf",
        }
        y = of_columns.get(of)
        columns = [
            "path_id",
            "booking_class",
            "days_prior",
            y,
        ]
        df = self.path_forecasts.reset_index()[columns]
        color = "booking_class:N"
        if isinstance(by_path_id, int) and by_path_id is not True:
            df = df[df.path_id == by_path_id]
        if isinstance(by_class, str):
            df = df[df.booking_class == by_class]
            color = None
        if raw_df:
            return df
        facet_on = None
        if by_path_id is True:
            facet_on = "path_id"
        return self._fig_forecasts(df, facet_on=facet_on, y=y, color=color)

    @report_figure
    def fig_bid_price_history(
        self,
        by_carrier: bool | str = True,
        show_stdev: float | bool | None = None,
        cap: Literal["some", "zero", None] = None,
        raw_df=False,
    ):
        if cap is None:
            bp_mean = "bid_price_mean"
        elif cap == "some":
            bp_mean = "some_cap_bid_price_mean"
        elif cap == "zero":
            bp_mean = "zero_cap_bid_price_mean"
        else:
            raise ValueError(f"cap={cap!r} not in ['some', 'zero', None]")
        df = self.bid_price_history.reset_index()
        color = None
        if isinstance(by_carrier, str):
            df = df[df.carrier == by_carrier]
        elif by_carrier:
            color = "carrier:N"
            if show_stdev is None:
                show_stdev = False
        if show_stdev:
            if show_stdev is True:
                show_stdev = 2
            df["bid_price_upper"] = df[bp_mean] + show_stdev * df["bid_price_stdev"]
            df["bid_price_lower"] = (
                df[bp_mean] - show_stdev * df["bid_price_stdev"]
            ).clip(0, None)
        if raw_df:
            return df

        import altair as alt

        line_encoding = dict(
            x=alt.X("days_prior:Q")
            .scale(reverse=True)
            .title("Days Prior to Departure"),
            y=alt.Y(bp_mean, title="Bid Price"),
        )
        if color:
            line_encoding["color"] = color
        chart = alt.Chart(df)
        fig = chart.mark_line(interpolate="step-before").encode(**line_encoding)
        if show_stdev:
            area_encoding = dict(
                x=alt.X("days_prior:Q")
                .scale(reverse=True)
                .title("Days Prior to Departure"),
                y=alt.Y("bid_price_lower:Q", title="Bid Price"),
                y2=alt.Y2("bid_price_upper:Q", title="Bid Price"),
            )
            bound = chart.mark_area(
                opacity=0.1,
                interpolate="step-before",
            ).encode(**area_encoding)
            bound_line = chart.mark_line(
                opacity=0.4, strokeDash=[5, 5], interpolate="step-before"
            ).encode(
                x=alt.X("days_prior:Q")
                .scale(reverse=True)
                .title("Days Prior to Departure")
            )
            top_line = bound_line.encode(
                y=alt.Y("bid_price_lower:Q", title="Bid Price")
            )
            bottom_line = bound_line.encode(
                y=alt.Y("bid_price_upper:Q", title="Bid Price")
            )
            fig = fig + bound + top_line + bottom_line
        return fig

    @report_figure
    def fig_displacement_history(
        self,
        by_carrier: bool | str = True,
        show_stdev: float | bool | None = None,
        raw_df=False,
    ):
        df = self.displacement_history.reset_index()
        color = None
        if isinstance(by_carrier, str):
            df = df[df.carrier == by_carrier]
        elif by_carrier:
            color = "carrier:N"
            if show_stdev is None:
                show_stdev = False
        if show_stdev:
            if show_stdev is True:
                show_stdev = 2
            df["displacement_upper"] = (
                df["displacement_mean"] + show_stdev * df["displacement_stdev"]
            )
            df["displacement_lower"] = (
                df["displacement_mean"] - show_stdev * df["displacement_stdev"]
            ).clip(0, None)
        if raw_df:
            return df

        import altair as alt

        line_encoding = dict(
            x=alt.X("days_prior:Q")
            .scale(reverse=True)
            .title("Days Prior to Departure"),
            y=alt.Y("displacement_mean", title="Displacement Cost"),
        )
        if color:
            line_encoding["color"] = color
        chart = alt.Chart(df)
        fig = chart.mark_line(interpolate="step-before").encode(**line_encoding)
        if show_stdev:
            area_encoding = dict(
                x=alt.X("days_prior:Q")
                .scale(reverse=True)
                .title("Days Prior to Departure"),
                y=alt.Y("displacement_lower:Q", title="Displacement Cost"),
                y2=alt.Y2("displacement_upper:Q", title="Displacement Cost"),
            )
            bound = chart.mark_area(
                opacity=0.1,
                interpolate="step-before",
            ).encode(**area_encoding)
            bound_line = chart.mark_line(
                opacity=0.4, strokeDash=[5, 5], interpolate="step-before"
            ).encode(
                x=alt.X("days_prior:Q")
                .scale(reverse=True)
                .title("Days Prior to Departure")
            )
            top_line = bound_line.encode(
                y=alt.Y("displacement_lower:Q", title="Displacement Cost")
            )
            bottom_line = bound_line.encode(
                y=alt.Y("displacement_upper:Q", title="Displacement Cost")
            )
            fig = fig + bound + top_line + bottom_line
        return fig
