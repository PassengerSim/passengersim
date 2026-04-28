from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import altair as alt

    from passengersim import Simulation, SimulationTables

from collections import defaultdict

import pandas as pd

from .generic import GenericTracer
from .welford import Welford


class _BidPriceTracer(GenericTracer):
    # name: str = "leg_bid_prices"
    bid_prices: dict[int, dict[int, Welford]]
    _on: str

    def __init__(self, _ids: list[int], *, priority: int = 1):
        self._ids = _ids
        self.priority = priority
        self.reset()

    def reset(self) -> None:
        self.bid_prices = {_id: defaultdict(Welford) for _id in self._ids}

    def fresh(self) -> _BidPriceTracer:
        return type(self)(self._ids, priority=self.priority)

    def attach(self, sim: Simulation) -> None:
        sim.daily_callback(self.fresh())

    def finalize(self) -> pd.DataFrame:
        m = pd.DataFrame(
            {
                _id: {days_prior: welford.mean for days_prior, welford in welfords.items()}
                for _id, welfords in self.bid_prices.items()
            }
        ).rename_axis(columns=self._on, index="days_prior")
        s = pd.DataFrame(
            {
                _id: {days_prior: welford.sample_std_dev for days_prior, welford in welfords.items()}
                for _id, welfords in self.bid_prices.items()
            }
        ).rename_axis(columns=self._on, index="days_prior")
        df = pd.concat([m, s], keys=["mean", "std_dev"], names=["statistic"])
        df.attrs["n_trials"] = 1
        df.attrs["aggregation_process"] = "wgt_by_trials"
        return df


class LegBidPriceTracer(_BidPriceTracer):
    name: str = "leg_bid_prices"
    _on: str = "leg_id"

    def __init__(self, leg_ids: list[int], *, priority: int = 1):
        super().__init__(leg_ids, priority=priority)

    def __call__(self, sim: Simulation, days_prior: int):
        if sim.eng.sample < sim.eng.burn_samples:
            return
        for leg_id in self._ids:
            self.bid_prices[leg_id][days_prior].update(sim.legs[leg_id].report_bid_price)


def _fig_traced_bid_prices(
    summary: SimulationTables,
    *,
    std_dev: bool = True,
    include: tuple[str] | None = None,
    raw_df: bool = False,
    kind: Literal["leg", "path"] = "leg",
    ids: tuple[int] | None = None,
) -> alt.Chart:
    from passengersim.contrast import Contrast  # prevent circular import

    if isinstance(summary, Contrast):
        if include is not None:
            summary = {k: v for k, v in summary.items() if k in include}
        dfs = {k: _fig_traced_bid_prices(v, raw_df=True, kind=kind) for k, v in summary.items()}
        df = pd.concat(dfs, names=["source"]).reset_index()
    else:
        df = getattr(summary.callback_data, f"{kind}_bid_prices")
        df = df.stack().unstack("statistic").reset_index()

    if ids is not None:
        df = df.query(f"{kind}_id in @ids")

    if raw_df:
        return df

    import altair as alt

    chart = alt.Chart(df)
    fig = chart.mark_line().encode(
        x=alt.X("days_prior:Q").scale(reverse=True),
        y=alt.Y("mean:Q", title="Mean Bid Price"),
        color=f"{kind}_id:N",
    )
    if "source" in df.columns:
        fig = fig.encode(strokeDash="source:N", strokeWidth="source:N")
    if std_dev:
        fig1 = chart.mark_line().encode(
            x=alt.X("days_prior:Q").scale(reverse=True),
            y=alt.Y("std_dev:Q", title="Std Dev Bid Price"),
            color=f"{kind}_id:N",
        )
        if "source" in df.columns:
            fig1 = fig1.encode(strokeDash="source:N", strokeWidth="source:N")
        fig = fig | fig1

    return fig


def fig_leg_bid_prices(
    summary: SimulationTables,
    *,
    std_dev: bool = True,
    include: tuple[str] | None = None,
    raw_df: bool = False,
    ids: tuple[int] | None = None,
) -> alt.Chart:
    return _fig_traced_bid_prices(summary, std_dev=std_dev, include=include, raw_df=raw_df, kind="leg", ids=ids)


class PathBidPriceTracer(_BidPriceTracer):
    """
    A daily tracer that records the bid prices for specified paths.
    """

    name: str = "path_bid_prices"
    _on: str = "path_id"

    def __init__(self, path_ids: list[int], *, priority=1):
        super().__init__(path_ids, priority=priority)

    def __call__(self, sim: Simulation, days_prior: int):
        if sim.eng.sample < sim.eng.burn_samples:
            return
        for path_id in self._ids:
            self.bid_prices[path_id][days_prior].update(sim.paths.select(path_id=path_id).report_total_bid_price())


def fig_path_bid_prices(
    summary: SimulationTables,
    *,
    std_dev: bool = True,
    include: tuple[str] | None = None,
    raw_df: bool = False,
    ids: tuple[int] | None = None,
) -> alt.Chart:
    return _fig_traced_bid_prices(summary, std_dev=std_dev, include=include, raw_df=raw_df, kind="path", ids=ids)


class LegBidPriceArrayTracer(GenericTracer):
    """
    A begin sample tracer that records the entire bid price arrays for specified legs.

    Note that as this is called at the beginning of the sample, it will not
    reflect any updates to bid prices that occur during the sample (e.g.,
    due to changes in displacement costs).
    """

    name: str = "leg_bid_price_arrays"
    bid_price_arrays: dict[int, Welford]

    def __init__(self, leg_ids: list[int], *, priority: int = 1):
        self.leg_ids = leg_ids
        self.priority = priority
        self.reset()

    def reset(self) -> None:
        self.bid_price_arrays = {leg_id: Welford() for leg_id in self.leg_ids}

    def fresh(self) -> LegBidPriceArrayTracer:
        return LegBidPriceArrayTracer(self.leg_ids, priority=self.priority)

    def attach(self, sim: Simulation) -> None:
        sim.begin_sample_callback(self.fresh())

    def __call__(self, sim: Simulation):
        if sim.eng.sample < sim.eng.burn_samples:
            return
        for leg_id in self.leg_ids:
            try:
                bpa = sim.legs[leg_id].get_bid_price_array()
            except ValueError:
                # If there is no bid price array (e.g., because not using bid prices), skip it
                continue
            else:
                self.bid_price_arrays[leg_id].update(bpa)

    def finalize(self) -> pd.DataFrame:
        data = {}
        for leg_id, w in self.bid_price_arrays.items():
            if w.n > 0:
                data[leg_id] = pd.DataFrame(
                    w.mean,
                    index=pd.RangeIndex(w.mean.shape[0], name="SeatsSold"),
                    columns=pd.RangeIndex(w.mean.shape[1], 0, step=-1, name="DaysPrior"),
                )
            else:
                data[leg_id] = pd.DataFrame(
                    index=pd.RangeIndex(0, name="SeatsSold"),
                    columns=pd.RangeIndex(0, name="DaysPrior"),
                )
        df = pd.concat(data, names=["leg_id"])
        df.attrs["n_trials"] = 1
        df.attrs["aggregation_process"] = "wgt_by_trials"
        return df


def fig_leg_bid_price_array(
    summary: SimulationTables,
    *,
    leg_id: int,
    raw_df: bool = False,
):
    df = summary.callback_data.leg_bid_price_arrays

    # select only the desired leg id
    df = df.query(f"leg_id == {leg_id}")

    # drop the leg id index level for plotting
    df = df.reset_index(level="leg_id", drop=True)

    max_index = df.index.max()
    if (df.loc[df.index.max()] == 0.0).all():
        # We expect the bid price array to be all zeros when all capacity is sold,
        # so we drop this row from the array
        df = df.drop(index=max_index)

    import altair as alt

    from passengersim.utils.heatmap import compress_and_tidy_heatmap_data

    tidy = compress_and_tidy_heatmap_data(df)
    tidy["DaysPrior2"] = tidy["DaysPrior"] - 1

    if raw_df:
        return tidy

    return (
        alt.Chart(tidy)
        .mark_rect()
        .encode(
            x=alt.X("DaysPrior:Q", title="Days Prior to Departure", scale=alt.Scale(reverse=True)),
            x2=alt.X2("DaysPrior2", title="Days Prior to Departure"),
            y=alt.Y("lowest_SeatsSold:Q", title="Seats Sold"),
            y2=alt.Y2("highest_SeatsSold:Q", title="Seats Sold"),
            color=alt.Color("bid_price:Q", title="Bid Price", scale=alt.Scale(scheme="viridis", domainMin=0)),
            tooltip=["DaysPrior", "SeatsSold", alt.Tooltip("bid_price", format="$,.2f", title="Bid Price")],
        )
        .interactive()
    )


class LegBidPriceVectorTracer(GenericTracer):
    """
    A daily tracer that records the entire bid price vector for specified legs.
    """

    name: str = "leg_bid_price_vectors"
    bid_price_vectors: dict[int, dict[int, Welford]]

    def __init__(self, leg_ids: list[int], *, priority: int = 2):
        self.leg_ids = leg_ids
        self.priority = priority
        self.reset()

    def reset(self) -> None:
        self.bid_price_vectors = {leg_id: defaultdict(Welford) for leg_id in self.leg_ids}

    def fresh(self) -> LegBidPriceVectorTracer:
        return LegBidPriceVectorTracer(self.leg_ids, priority=self.priority)

    def attach(self, sim: Simulation) -> None:
        sim.daily_callback(self.fresh())

    def __call__(self, sim: Simulation, days_prior: int):
        if sim.eng.sample < sim.eng.burn_samples:
            return
        if days_prior == 0:
            # do not store bid price vectors at the moment of departure, they are meaningless
            return
        for leg_id in self.leg_ids:
            try:
                bpa = sim.legs[leg_id].get_bid_price_array()
            except ValueError:
                # If there is no bid price array (e.g., because not using bid prices), skip it
                continue
            else:
                if bpa.ndim > 1 and bpa.shape[1] > 1:
                    # if the bid price array is a full array, select this day
                    bpa = bpa[:, sim.legs[leg_id].bp_index]
                self.bid_price_vectors[leg_id][days_prior].update(bpa)

    def finalize(self) -> pd.DataFrame:
        data = {}
        for leg_id in self.bid_price_vectors.keys():
            subdata = {}
            for days_prior, b in self.bid_price_vectors[leg_id].items():
                if b.n > 0:
                    subdata[days_prior] = pd.DataFrame(
                        b.mean,
                        index=pd.RangeIndex(b.mean.shape[0], name="SeatsSold"),
                    )
                else:
                    subdata[days_prior] = pd.DataFrame(
                        index=pd.RangeIndex(0, name="SeatsSold"),
                        columns=pd.RangeIndex(0, name="DaysPrior"),
                    )
            if subdata:
                data[leg_id] = pd.concat(subdata, names=["days_prior"])
        if data:
            df = pd.concat(data, names=["leg_id"])
        else:
            df = pd.DataFrame()
        df.attrs["n_trials"] = 1
        df.attrs["aggregation_process"] = "wgt_by_trials"
        return df


def fig_leg_bid_price_vectors(
    summary: SimulationTables,
    *,
    leg_id: int,
    raw_df: bool = False,
):
    df = summary.callback_data.leg_bid_price_vectors

    # select only the desired leg id
    df = df.query(f"leg_id == {leg_id}")

    # drop the leg id index level for plotting
    df = df.reset_index(level="leg_id", drop=True)

    # convert to bid price array
    df = df.iloc[:, 0].unstack("days_prior")

    max_index = df.index.max()
    if (df.loc[df.index.max()] == 0.0).all():
        # We expect the bid price array to be all zeros when all capacity is sold,
        # so we drop this row from the array
        df = df.drop(index=max_index)

    import altair as alt

    from passengersim.utils.heatmap import compress_and_tidy_heatmap_data

    tidy = compress_and_tidy_heatmap_data(df)
    tidy["DaysPrior2"] = tidy["days_prior"] - 1

    if raw_df:
        return tidy

    return (
        alt.Chart(tidy)
        .mark_rect()
        .encode(
            x=alt.X("days_prior:Q", title="Days Prior to Departure", scale=alt.Scale(reverse=True)),
            x2=alt.X2("DaysPrior2", title="Days Prior to Departure"),
            y=alt.Y("lowest_SeatsSold:Q", title="Seats Sold"),
            y2=alt.Y2("highest_SeatsSold:Q", title="Seats Sold"),
            color=alt.Color("bid_price:Q", title="Bid Price", scale=alt.Scale(scheme="viridis", domainMin=0)),
            tooltip=[
                alt.Tooltip("days_prior", title="Days Prior"),
                alt.Tooltip("SeatsSold", title="Seats Sold"),
                alt.Tooltip("bid_price", format="$,.2f", title="Bid Price"),
            ],
        )
        .interactive()
    )
