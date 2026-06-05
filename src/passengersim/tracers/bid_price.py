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
    """Private base class shared by bid-price tracers.

    Collects per-sample bid prices keyed by an entity ID and days-prior,
    then summarises them with a :class:`Welford` online mean/variance
    accumulator.  Concrete subclasses must set the class attribute ``_on``
    to the column name used for the entity dimension (e.g. ``"leg_id"`` or
    ``"path_id"``).
    """

    # name: str = "leg_bid_prices"
    bid_prices: dict[int, dict[int, Welford]]
    """Nested mapping of entity-ID → days-prior → Welford accumulator."""

    _on: str
    """Name of the entity-dimension column produced by :meth:`finalize`."""

    def __init__(self, _ids: list[int], *, priority: int = 1) -> None:
        """Initialize the tracer.

        Parameters
        ----------
        _ids : list[int]
            Entity IDs (leg IDs or path IDs) to track.
        priority : int, optional
            Callback priority.  Lower values run first.  Defaults to 1.
        """
        self._ids = _ids
        self.priority = priority
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated bid-price data to empty Welford accumulators."""
        self.bid_prices = {_id: defaultdict(Welford) for _id in self._ids}

    def fresh(self) -> _BidPriceTracer:
        """Return a new, empty tracer of the same type and configuration.

        Returns
        -------
        _BidPriceTracer
            A freshly initialized tracer of the same concrete type, with the
            same IDs and priority but no accumulated data.
        """
        return type(self)(self._ids, priority=self.priority)

    def attach(self, sim: Simulation) -> None:
        """Attach a fresh instance of this tracer to *sim* as a daily callback.

        Parameters
        ----------
        sim : Simulation
            The simulation to attach the callback to.
        """
        sim.daily_callback(self.fresh())

    def finalize(self) -> pd.DataFrame:
        """Aggregate accumulated Welford statistics into a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame with a MultiIndex of ``(statistic, days_prior)``
            containing mean and sample standard deviation bid prices for each
            tracked entity.  The DataFrame attrs ``n_trials`` and
            ``aggregation_process`` are set for downstream aggregation.
        """
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
    """A daily tracer that records bid prices for specified legs.

    At each daily data collection, the current bid price for each
    tracked leg is captured and accumulated using a Welford online estimator
    so that the mean and standard deviation across simulation samples can be
    computed efficiently.
    """

    name: str = "leg_bid_prices"
    """Identifier used to store results in :attr:`SimulationTables.callback_data`."""

    _on: str = "leg_id"
    """Column name for the leg dimension in the output DataFrame."""

    def __init__(self, leg_ids: list[int], *, priority: int = 1) -> None:
        """Initialize the tracer.

        Parameters
        ----------
        leg_ids : list[int]
            IDs of the legs whose bid prices should be tracked.
        priority : int, optional
            Callback priority.  Lower values run first.  Defaults to 1.
        """
        super().__init__(leg_ids, priority=priority)

    def __call__(self, sim: Simulation, days_prior: int) -> None:
        """Record the current bid price for each tracked leg.

        Skips burn-in samples.

        Parameters
        ----------
        sim : Simulation
            The running simulation.
        days_prior : int
            Days remaining before departure at the time of this callback.
        """
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
) -> alt.Chart | pd.DataFrame:
    """Build a bid-price chart (or raw DataFrame) for legs or paths.

    Parameters
    ----------
    summary : SimulationTables or Contrast
        Simulation results to plot.  A :class:`~passengersim.contrast.Contrast`
        mapping is also accepted, in which case each entry is plotted with a
        distinct stroke style.
    std_dev : bool, optional
        When ``True`` (default), a second panel showing the standard deviation
        is placed beside the mean panel.
    include : tuple[str] | None, optional
        When *summary* is a :class:`~passengersim.contrast.Contrast`, restrict
        the plot to only the keys listed here.  Ignored for a single
        ``SimulationTables``.
    raw_df : bool, optional
        When ``True``, return the tidy :class:`pandas.DataFrame` instead of an
        Altair chart.  Defaults to ``False``.
    kind : {"leg", "path"}, optional
        Whether to plot leg bid prices (``"leg"``) or path bid prices
        (``"path"``).  Defaults to ``"leg"``.
    ids : tuple[int] | None, optional
        Restrict the plot to the entity IDs listed here.  When ``None``
        (default) all tracked entities are shown.

    Returns
    -------
    alt.Chart or pd.DataFrame
        An Altair chart when *raw_df* is ``False``, or a
        :class:`pandas.DataFrame` when *raw_df* is ``True``.
    """
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
) -> alt.Chart | pd.DataFrame:
    """Plot mean (and optionally standard-deviation) bid prices for tracked legs.

    Parameters
    ----------
    summary : SimulationTables or Contrast
        Simulation results to plot.  A :class:`~passengersim.contrast.Contrast`
        mapping is also accepted.
    std_dev : bool, optional
        When ``True`` (default), a second panel showing the standard deviation
        is placed beside the mean panel.
    include : tuple[str] | None, optional
        When *summary* is a :class:`~passengersim.contrast.Contrast`, restrict
        the plot to only the keys listed here.
    raw_df : bool, optional
        When ``True``, return the tidy :class:`pandas.DataFrame` instead of an
        Altair chart.  Defaults to ``False``.
    ids : tuple[int] | None, optional
        Restrict the plot to the leg IDs listed here.  When ``None`` (default)
        all tracked legs are shown.

    Returns
    -------
    alt.Chart or pd.DataFrame
        An Altair chart when *raw_df* is ``False``, or a
        :class:`pandas.DataFrame` when *raw_df* is ``True``.
    """
    return _fig_traced_bid_prices(summary, std_dev=std_dev, include=include, raw_df=raw_df, kind="leg", ids=ids)


class PathBidPriceTracer(_BidPriceTracer):
    """A daily tracer that records the bid prices for specified paths.

    At each day, the total bid price for each tracked path is captured and accumulated
    using a Welford online estimator, so that the mean and standard deviation across
    simulation samples can be computed efficiently.
    """

    name: str = "path_bid_prices"
    """Identifier used to store results in :attr:`SimulationTables.callback_data`."""

    _on: str = "path_id"
    """Column name for the path dimension in the output DataFrame."""

    def __init__(self, path_ids: list[int], *, priority: int = 1) -> None:
        """Initialize the tracer.

        Parameters
        ----------
        path_ids : list[int]
            IDs of the paths whose bid prices should be tracked.
        priority : int, optional
            Callback priority.  Lower values run first.  Defaults to 1.
        """
        super().__init__(path_ids, priority=priority)

    def __call__(self, sim: Simulation, days_prior: int) -> None:
        """Record the current total bid price for each tracked path.

        Skips burn-in samples.

        Parameters
        ----------
        sim : Simulation
            The running simulation.
        days_prior : int
            Days remaining before departure at the time of this callback.
        """
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
) -> alt.Chart | pd.DataFrame:
    """Plot mean (and optionally standard-deviation) bid prices for tracked paths.

    Parameters
    ----------
    summary : SimulationTables or Contrast
        Simulation results to plot.  A :class:`~passengersim.contrast.Contrast`
        mapping is also accepted.
    std_dev : bool, optional
        When ``True`` (default), a second panel showing the standard deviation
        is placed beside the mean panel.
    include : tuple[str] | None, optional
        When *summary* is a :class:`~passengersim.contrast.Contrast`, restrict
        the plot to only the keys listed here.
    raw_df : bool, optional
        When ``True``, return the tidy :class:`pandas.DataFrame` instead of an
        Altair chart.  Defaults to ``False``.
    ids : tuple[int] | None, optional
        Restrict the plot to the path IDs listed here.  When ``None`` (default)
        all tracked paths are shown.

    Returns
    -------
    alt.Chart or pd.DataFrame
        An Altair chart when *raw_df* is ``False``, or a
        :class:`pandas.DataFrame` when *raw_df* is ``True``.
    """
    return _fig_traced_bid_prices(summary, std_dev=std_dev, include=include, raw_df=raw_df, kind="path", ids=ids)


class LegBidPriceArrayTracer(GenericTracer):
    """A begin-sample tracer that records the entire bid-price arrays for specified legs.

    The full bid-price array (seats-sold × days-prior) is captured at the
    start of each simulation sample and accumulated using a Welford online
    estimator.  Note that because this tracer is called at the *beginning* of
    each sample, it does not reflect bid-price updates that occur during the
    sample (e.g., due to changes in displacement costs).
    """

    name: str = "leg_bid_price_arrays"
    """Identifier used to store results in :attr:`SimulationTables.callback_data`."""

    bid_price_arrays: dict[int, Welford]
    """Mapping of leg-ID → Welford accumulator over the full bid-price array."""

    def __init__(self, leg_ids: list[int], *, priority: int = 1) -> None:
        """Initialize the tracer.

        Parameters
        ----------
        leg_ids : list[int]
            IDs of the legs whose bid-price arrays should be tracked.
        priority : int, optional
            Callback priority.  Lower values run first.  Defaults to 1.
        """
        self.leg_ids = leg_ids
        self.priority = priority
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated bid-price array data to empty Welford accumulators."""
        self.bid_price_arrays = {leg_id: Welford() for leg_id in self.leg_ids}

    def fresh(self) -> LegBidPriceArrayTracer:
        """Return a new, empty tracer of the same type and configuration.

        Returns
        -------
        LegBidPriceArrayTracer
            A freshly initialised tracer with the same leg IDs and priority
            but no accumulated data.
        """
        return LegBidPriceArrayTracer(self.leg_ids, priority=self.priority)

    def attach(self, sim: Simulation) -> None:
        """Attach a fresh instance of this tracer to *sim* as a begin-sample callback.

        Parameters
        ----------
        sim : Simulation
            The simulation to attach the callback to.
        """
        sim.begin_sample_callback(self.fresh())

    def __call__(self, sim: Simulation) -> None:
        """Capture the bid-price array for each tracked leg at the start of a sample.

        Skips burn-in samples and silently skips any leg that does not have a
        bid-price array (e.g., because bid prices are not enabled for that leg).

        Parameters
        ----------
        sim : Simulation
            The running simulation.
        """
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
        """Aggregate accumulated Welford statistics into a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by ``(leg_id, SeatsSold)`` with columns
            representing days-prior values, containing the mean bid price at
            each cell.  The DataFrame attrs ``n_trials`` and
            ``aggregation_process`` are set for downstream aggregation.
        """
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
) -> alt.Chart | pd.DataFrame:
    """Plot the mean bid-price array heatmap for a single leg.

    Parameters
    ----------
    summary : SimulationTables
        Simulation results containing
        :attr:`~SimulationTables.callback_data.leg_bid_price_arrays`.
    leg_id : int
        The ID of the leg to plot.
    raw_df : bool, optional
        When ``True``, return the tidy :class:`pandas.DataFrame` used to build
        the chart instead of the chart itself.  Defaults to ``False``.

    Returns
    -------
    alt.Chart or pd.DataFrame
        An interactive Altair heatmap chart when *raw_df* is ``False``, or a
        :class:`pandas.DataFrame` when *raw_df* is ``True``.
    """
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
    """A daily tracer that records the entire bid-price vector for specified legs.

    At each day prior to departure, the current bid-price vector
    (one entry per seats-sold level) is captured for each tracked leg and
    accumulated using a Welford online estimator.
    """

    name: str = "leg_bid_price_vectors"
    """Identifier used to store results in :attr:`SimulationTables.callback_data`."""

    bid_price_vectors: dict[int, dict[int, Welford]]
    """Nested mapping of leg-ID → days-prior → Welford accumulator over the bid-price vector."""

    def __init__(self, leg_ids: list[int], *, priority: int = 2) -> None:
        """Initialize the tracer.

        Parameters
        ----------
        leg_ids : list[int]
            IDs of the legs whose bid-price vectors should be tracked.
        priority : int, optional
            Callback priority.  Lower values run first.  Defaults to 2.
        """
        self.leg_ids = leg_ids
        self.priority = priority
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated bid-price vector data to empty Welford accumulators."""
        self.bid_price_vectors = {leg_id: defaultdict(Welford) for leg_id in self.leg_ids}

    def fresh(self) -> LegBidPriceVectorTracer:
        """Return a new, empty tracer of the same type and configuration.

        Returns
        -------
        LegBidPriceVectorTracer
            A freshly initialised tracer with the same leg IDs and priority
            but no accumulated data.
        """
        return LegBidPriceVectorTracer(self.leg_ids, priority=self.priority)

    def attach(self, sim: Simulation) -> None:
        """Attach a fresh instance of this tracer to *sim* as a daily callback.

        Parameters
        ----------
        sim : Simulation
            The simulation to attach the callback to.
        """
        sim.daily_callback(self.fresh())

    def __call__(self, sim: Simulation, days_prior: int) -> None:
        """Record the current bid-price vector for each tracked leg.

        Skips burn-in samples, skips day zero (bid prices are meaningless at
        departure), and silently skips any leg without a bid-price array.

        Parameters
        ----------
        sim : Simulation
            The running simulation.
        days_prior : int
            Days remaining before departure at the time of this callback.
        """
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
        """Aggregate accumulated Welford statistics into a tidy DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by ``(leg_id, days_prior, SeatsSold)`` with a
            single column containing the mean bid price at each cell.  The
            DataFrame attrs ``n_trials`` and ``aggregation_process`` are set
            for downstream aggregation.
        """
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
) -> alt.Chart | pd.DataFrame:
    """Plot the mean bid-price vector heatmap for a single leg.

    Parameters
    ----------
    summary : SimulationTables
        Simulation results containing
        :attr:`~SimulationTables.callback_data.leg_bid_price_vectors`.
    leg_id : int
        The ID of the leg to plot.
    raw_df : bool, optional
        When ``True``, return the tidy :class:`pandas.DataFrame` used to build
        the chart instead of the chart itself.  Defaults to ``False``.

    Returns
    -------
    alt.Chart or pd.DataFrame
        An interactive Altair heatmap chart when *raw_df* is ``False``, or a
        :class:`pandas.DataFrame` when *raw_df* is ``True``.
    """
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
