import logging
import warnings
from collections import namedtuple

import altair as alt
import numpy as np
import pandas as pd

from passengersim import Config, Simulation
from passengersim.config.checks.markets import check_min_fare_price_by_market
from passengersim.core import Generator
from passengersim.driver._constructors import make_core_booking_curve, make_core_choice_model, make_core_demand
from passengersim.driver._demand_gen import generate_sample_demands
from passengersim.tracers.welford import SingleWelford, Welford
from passengersim.utils.string_counting import StringTracker

DemandGenChecks = namedtuple(
    "DemandGenChecks",
    [
        "orig",
        "dest",
        "reference_price",
        "emults",
        "welford",
        "wtp_summary",
        "wtp_vectors",
        "wtp_pct_greater",
        "segments",
        "segment_stats",
        "frat_5_estimates",
        "frat_5_estimates_ap",
    ],
)


def _get_wtpQuantile_for_price(wtp_vector, price):
    """Return the fraction of passengers whose maximum WTP is at least *price*.

    Parameters
    ----------
    wtp_vector : array-like
        Sorted or unsorted array of individual maximum willingness-to-pay values.
    price : float
        The price threshold to evaluate against.

    Returns
    -------
    float
        Fraction in [0, 1] of passengers willing to pay at least *price*.
    """
    return np.mean(wtp_vector >= price)


def _get_price_for_wtpQuantile(wtp_vector, quantile):
    """Return the price at which exactly *quantile* fraction of passengers are willing to pay.

    Parameters
    ----------
    wtp_vector : array-like
        Array of individual maximum willingness-to-pay values.
    quantile : float
        Target fraction of passengers, in (0, 1].  For example, 0.5 returns the
        median WTP.

    Returns
    -------
    float
        Price point such that the given fraction of passengers have WTP ≥ that price.
    """
    return np.quantile(wtp_vector, 1 - quantile)


def _get_frat5_at_price(wtp_vector, starting_price):
    """Return the fare ratio at which half of the passengers willing to pay *starting_price* will disappear.

    This is the "Frat5" metric: the ratio of the price at which only half the
    original willing passengers remain, divided by *starting_price*.

    Parameters
    ----------
    wtp_vector : array-like
        Array of individual maximum willingness-to-pay values.
    starting_price : float
        The reference price from which the Frat5 ratio is calculated.

    Returns
    -------
    float
        Frat5 ratio, or ``np.nan`` if no passengers are willing to pay
        *starting_price* (making the ratio undefined).
    """
    q = _get_wtpQuantile_for_price(wtp_vector, starting_price)
    if q == 0:
        # If no passengers are willing to pay the starting price,
        # the fare ratio at which half will disappear is infinite.
        # We will return NaN instead of infinity to indicate that this is not a meaningful value.
        return np.nan
    else:
        target_quantile = q / 2
        price_at_target_quantile = _get_price_for_wtpQuantile(wtp_vector, target_quantile)
        frat5 = price_at_target_quantile / starting_price
        return frat5


def check_demand_generation(
    cfg: Config,
    orig: str,
    dest: str,
    carrier: str | None = None,
    *,
    n_samples: int = 3653,
    n_draws_per_tf: int = 100_000,
    wtp_resolution: int = 200,
    raw_data: bool = False,
):
    """Check the random demand generation for a single market.

    Runs *n_samples* Monte-Carlo demand draws for the given origin–destination
    market, allocates the resulting passengers across booking timeframes, and
    summarizes the willingness-to-pay (WTP) distribution for each timeframe.
    The results are returned either as a structured :class:`DemandGenChecks`
    named-tuple (``raw_data=True``) or as a multi-panel Altair dashboard.

    The panels in the dashboard include:
    - A set of willingness-to-pay survival curves showing the fraction of
      passengers with WTP above each price point, for each booking timeframe.
    - WTP statistics showing the mean, median, and interquartile range of
      maximum WTP (this is the same data as the first panel in a different
      visualization format).
    - Segment-level WTP statistics showing the mean WTP relative to the
      "reference price" for each passenger segment. The thin white lines show
      the reference price inside each segment, while the overall bars show
      the mean WTP.
    - Mean demand arriving in each timeframe, by segment.
    - Unconditional Frat5 curves, showing for each booking class the fare ratio
      at which half of the customers who are willing to pay that fare will
      no longer be willing to pay. This value varies by fare level and timeframe,
      and is not directly reflective of a "Frat5" in application because it
      does not consider any competitive effects.  It does however offer an idea
      of an upper bound on the practical Frat5.

    Parameters
    ----------
    cfg : Config
        Simulation configuration.  The function operates on a deep copy so the
        caller's object is never mutated.
    orig : str
        IATA origin airport code for the market to analyse.
    dest : str
        IATA destination airport code for the market to analyse.
    carrier : str or None, optional
        If provided, restrict the fare set to a single carrier.  Defaults to
        ``None`` (all carriers in the market are included).
    n_samples : int, optional
        Number of demand-generation Monte-Carlo samples.  Defaults to 3653
        (approximately ten years of daily draws).
    n_draws_per_tf : int, optional
        Number of individual passenger draws to generate in each booking
        timeframe.  Defaults to 100,000.
    wtp_resolution : int, optional
        Number of price points used to build the WTP survival curves.
        Defaults to 200.
    raw_data : bool, optional
        When ``True``, return the raw :class:`DemandGenChecks` named-tuple
        instead of the visualisation dashboard.  Defaults to ``False``.

    Returns
    -------
    alt.VConcatChart or DemandGenChecks
        An Altair dashboard when ``raw_data=False`` (the default), or a
        :class:`DemandGenChecks` named-tuple when ``raw_data=True``.

    Raises
    ------
    ValueError
        If no demand configurations are found for the requested market.
    """

    # ensure the market given has some demand configs
    demands = [d for d in cfg.demands if d.orig == orig and d.dest == dest]
    if len(demands) == 0:
        raise ValueError(f"No demand configs found for market {orig}~{dest}")

    # set the "general" reference price as the lowest reference price among the demand configs for this market
    reference_prices = {
        d.segment: (d.reference_price * cfg.choice_models[d.choice_model or d.segment].reference_price_multiplier)
        for d in demands
    }

    # collect emults from each demand, preferring the emult defined on the demand itself if it exists,
    # and falling back to the emult defined on the choice model if not
    emults = {}
    for d in demands:
        if d.emult is not None:
            emults[d.segment] = d.emult
        else:
            emults[d.segment] = cfg.choice_models[d.choice_model_].emult

    # don't manipulate the config you were given, make a copy
    cfg = cfg.model_copy(deep=True)
    cfg.simulation_controls.allow_unused_restrictions = True

    cfg.demands = demands
    if carrier is None:
        cfg.fares = [f for f in cfg.fares if f.orig == orig and f.dest == dest]
    else:
        cfg.fares = [f for f in cfg.fares if f.orig == orig and f.dest == dest and f.carrier == carrier]
    cfg.legs = []
    cfg.paths = []
    cfg.simulation_controls.connection_builder.existing_paths = "keep"

    with warnings.catch_warnings(record=False):
        warnings.simplefilter("ignore")
        sim = Simulation(cfg)
    end_time = sim.eng.base_time

    n_demands = len(cfg.demands)
    segments = [d.segment for d in cfg.demands]
    n_tf = len(cfg.dcps)

    counter_array = np.zeros([n_demands, n_tf], dtype=np.int32)
    welford = Welford()
    welford_by_segments = {s: SingleWelford() for s in segments}

    for _i in range(n_samples):
        generate_sample_demands(
            sim.eng,
            cfg.simulation_controls,
            allocate=False,
        )
        for n, dmd in enumerate(sim.demands):
            num_pax = int(dmd.scenario_demand + 0.5)  # rounding
            dmd_by_tf = sim.eng.allocate_demand_to_tf(
                dmd, num_pax=num_pax, tf_k_factor=cfg.simulation_controls.tf_k_factor, end_ts=end_time
            )
            counter_array[n, :] = dmd_by_tf
        welford.update(counter_array)
        sim.eng._purge_event_queue()

    tf_boost_factor = (n_draws_per_tf // (welford.mean * welford.n).astype(int).sum(0)) + 1
    draw_table = (welford.mean * welford.n).astype(int) * tf_boost_factor
    wtp_vectors = {}
    wtp_pct_greater = np.empty([n_tf, wtp_resolution], dtype=np.float32)

    max_fare_price = max([f.price for f in cfg.fares])
    lower_bound = 0
    upper_bound = max_fare_price * 6.0
    x_values = np.linspace(lower_bound, upper_bound, wtp_resolution * 5)

    for tf in range(n_tf):
        tf_draws = draw_table[:, tf].sum(0)
        wtp = np.empty(tf_draws, dtype=np.float32)
        z = 0
        for n, dmd in enumerate(sim.demands):
            cm = dmd.choice_model
            dmd_n_draws = draw_table[n, tf]
            if dmd_n_draws > 0:
                dmd_wtp = cm.max_wtp(dmd.reference_price, n_draws=dmd_n_draws, raw=True, emult=dmd.emult)
                welford_by_segments[dmd.segment].update(dmd_wtp["raw"])
                wtp[z : z + dmd_n_draws] = dmd_wtp["raw"]
                z += dmd_n_draws
        wtp_vectors[tf] = np.sort(wtp)
        wtp_pct_greater[tf, :] = 100.0 * (
            1 - np.searchsorted(wtp_vectors[tf], x_values[:wtp_resolution], side="left") / wtp_vectors[tf].size
        )

    wtp_summary = pd.DataFrame({"dcp": cfg.dcps}, index=pd.Index(range(n_tf), name="tf"))
    wtp_summary["mean"] = {k: np.mean(v) for k, v in wtp_vectors.items()}
    wtp_summary["std"] = {k: np.std(v) for k, v in wtp_vectors.items()}

    wtp_pct_greater = pd.DataFrame(
        wtp_pct_greater,
        index=pd.Index(range(n_tf), name="tf"),
        columns=pd.Index(x_values[:wtp_resolution], name="price_point"),
    )

    quantiles = {}
    for tf in range(n_tf):
        quantiles[tf] = pd.Series(
            np.quantile(wtp_vectors[tf], [0.05, 0.10, 0.125, 0.25, 0.375, 0.50, 0.75, 0.90, 0.99]),
            index=["q05", "q10", "q12", "q25", "q37", "q50", "q75", "q90", "q99"],
        )

    quantiles_df = pd.concat(quantiles, axis=1, names=["tf"])
    quantiles_df.index.name = "quantile"

    wtp_summary = pd.concat([wtp_summary, quantiles_df.T], axis=1)
    wtp_summary = wtp_summary.eval("frat5_99 = q50 / q99")
    wtp_summary = wtp_summary.eval("frat5_75 = q37 / q75")
    wtp_summary = wtp_summary.eval("frat5_50 = q25 / q50")
    wtp_summary = wtp_summary.eval("frat5_25 = q12 / q25")
    wtp_summary = wtp_summary.eval("frat5_10 = q05 / q10")

    segment_stats = (
        pd.DataFrame(
            {s: {"mean": welford_by_segments[s].mean, "std": welford_by_segments[s].std_dev} for s in segments}
        )
        .T.rename_axis(index="segment")
        .sort_values(by="mean", ascending=False)
    )

    frat_5_estimates = {}
    fare_levels = {f.booking_class: f.price for f in cfg.fares}
    for bookclass, price in fare_levels.items():
        frat_5_estimates[(bookclass, price)] = [_get_frat5_at_price(wtp_vectors[tf], price) for tf in range(n_tf)]
    frat_5_estimates = pd.DataFrame.from_dict(frat_5_estimates).rename_axis(
        columns=["booking_class", "price"], index="tf"
    )

    np.searchsorted(cfg.dcps[::-1], 0, side="right")
    unavailable_tfs = {
        f.booking_class: int(np.searchsorted(cfg.dcps[::-1], f.advance_purchase, side="right")) for f in cfg.fares
    }

    # create a copy of frat_5_estimates where places closed by AP are removed
    frat_5_estimates_ap = frat_5_estimates.copy()
    for bc, nan_tfs in unavailable_tfs.items():
        if nan_tfs:
            frat_5_estimates_ap.loc[frat_5_estimates_ap.index[-nan_tfs:], bc] = np.nan

    # put segments in order of mean WTP, highest first
    segments = segment_stats.index.tolist()

    output = DemandGenChecks(
        orig,
        dest,
        reference_prices,
        emults,
        welford,
        wtp_summary,
        wtp_vectors,
        wtp_pct_greater,
        segments,
        segment_stats,
        frat_5_estimates,
        frat_5_estimates_ap,
    )

    if raw_data:
        return output
    return _viz_check_demand_generation(output)


def _viz_frat5(raw_data: DemandGenChecks, height: int, width: int) -> alt.LayerChart:
    """Build the Unconditional Frat5 panel.

    Renders one line per booking class showing how the Frat5 metric (the fare
    ratio at which half the originally-willing passengers disappear) evolves
    across booking timeframes.  An interactive hover rule displays exact values
    for all booking classes at the selected timeframe.

    Parameters
    ----------
    raw_data : DemandGenChecks
        Aggregated demand-generation check data.
    height : int
        Panel height in pixels.
    width : int
        Panel width in pixels.

    Returns
    -------
    alt.LayerChart
        Layered Altair chart (lines + points + hover rule).
    """
    df_base = (
        pd.DataFrame.from_dict(raw_data.frat_5_estimates)
        .rename_axis(columns=["booking_class", "price"], index="tf")
        .T.stack()
        .rename("frat5value")
    )
    df_ap = (
        pd.DataFrame.from_dict(raw_data.frat_5_estimates_ap)
        .rename_axis(columns=["booking_class", "price"], index="tf")
        .T.stack()
        .rename("frat5value_ap")
    )
    df_base = pd.concat([df_base, df_ap], axis=1)
    df = df_base.reset_index().eval("tf = tf + 1")
    df["frat5_off"] = df["frat5value"]
    _off = ~(df.groupby("booking_class")["frat5value_ap"].shift(-1).isnull())
    df.loc[_off, "frat5_off"] = np.nan

    df["label"] = df.booking_class + df["price"].map(lambda x: f" (${x:,.0f})")

    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["tf"], empty=False)
    when_near = alt.when(nearest)

    _frat5chart = alt.Chart(
        df,
        width=width,
        height=height,
        title=alt.TitleParams("Unconditional Frat5", anchor="middle", fontSize=14),
    )

    _frat5lines = _frat5chart.mark_line().encode(
        x=alt.X("tf:O", title="Timeframe", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("frat5value_ap", title="Frat5 Value", scale=alt.Scale(zero=False)),
        color=alt.Color("label:N", title="Booking Class"),
    )

    _frat5offlines = _frat5chart.mark_line(strokeDash=[2, 4], strokeWidth=1).encode(
        x=alt.X("tf:O", title="Timeframe", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("frat5_off", title="Frat5 Value", scale=alt.Scale(zero=False)),
        color=alt.Color("label:N", title="Booking Class"),
    )

    # Draw points on the line, and highlight based on selection
    _frat5points = _frat5lines.mark_point().encode(
        opacity=when_near.then(alt.value(1)).otherwise(alt.value(0)),
        y=alt.Y("frat5value", title="Frat5 Value", scale=alt.Scale(zero=False)),
    )

    _frat5rules = (
        alt.Chart(df)
        .transform_pivot("label", value="frat5value", groupby=["tf"])
        .mark_rule(color="gray")
        .encode(
            x=alt.X("tf:O", axis=alt.Axis(labelAngle=0)),
            opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
            tooltip=[alt.Tooltip("tf", title="Timeframe")]
            + [alt.Tooltip(c, type="quantitative", format=".3f") for c in df.label.unique()],
        )
        .add_params(nearest)
    )

    return alt.layer(_frat5lines, _frat5offlines, _frat5points, _frat5rules)


def _viz_segment_stats(raw_data: DemandGenChecks, height: int, width: int) -> alt.LayerChart:
    """Build the Segment Stats panel.

    Renders a bar chart of mean WTP per passenger segment, overlaid with a
    white error bar that spans from zero to the segment's reference price,
    giving a visual comparison between simulated WTP and the configured
    reference price.

    Parameters
    ----------
    raw_data : DemandGenChecks
        Aggregated demand-generation check data.
    height : int
        Panel height in pixels.
    width : int
        Panel width in pixels.

    Returns
    -------
    alt.LayerChart
        Layered Altair chart (bars + reference-price error bar).
    """
    segment_stats = raw_data.segment_stats
    segments = raw_data.segments
    ref_prices = raw_data.reference_price
    emults = raw_data.emults

    df = (
        segment_stats.join(pd.Series(ref_prices, name="reference_price"))
        .join(pd.Series(emults, name="emult"))
        .reset_index()
        .eval("zero=0")
    )
    df["mean_v_ref"] = df["mean"] / df["reference_price"]
    df["std_v_ref"] = df["std"] / df["reference_price"]

    _segment_stats_chart = alt.Chart(
        df,
        title=alt.TitleParams("Segment Stats", anchor="middle", fontSize=14),
        width=width,
        height=height,
    )

    tooltips = [
        alt.Tooltip("segment:N", title="Segment"),
        alt.Tooltip("mean:Q", title="Mean WTP", format="$.2f"),
        alt.Tooltip("std:Q", title="Std Dev WTP", format="$.2f"),
        alt.Tooltip("mean_v_ref:Q", title="Mean vs Ref Price", format=".2f"),
        alt.Tooltip("std_v_ref:Q", title="Std Dev vs Ref Price", format=".2f"),
        alt.Tooltip("reference_price:Q", title="Ref Price", format="$.2f"),
        alt.Tooltip("emult:Q", title="Emult", format=".2f"),
    ]

    segment_stats_chart = _segment_stats_chart.mark_bar().encode(
        x=alt.X("segment:N", title="Segment", axis=alt.Axis(labelAngle=0), sort=segments),
        y=alt.Y("mean:Q", title="Willingness to Pay", axis=alt.Axis(format="$.0f")),
        color=alt.Color("segment:N", title="Segment", sort=segments),
        tooltip=tooltips,
    ) + _segment_stats_chart.mark_errorbar(color="#ffffff", thickness=2).encode(
        x=alt.X("segment:N", title="Segment", axis=alt.Axis(labelAngle=0), sort=segments),
        y=alt.Y("reference_price:Q", title="Willingness to Pay"),
        y2=alt.Y2("zero:Q"),
        tooltip=tooltips,
    )

    return segment_stats_chart


def _viz_wtp_survival_curves(
    raw_data: DemandGenChecks,
    width: int = 250,
    height: int = 250,
) -> alt.TopLevelMixin:
    """Build the Willingness-to-Pay Survival Curves panel.

    Each line represents one booking timeframe and shows the fraction of
    passengers whose maximum WTP exceeds a given price point (the "survival"
    curve of the WTP distribution).  An interactive hover rule displays exact
    percentages for all timeframes at the selected price point.

    Parameters
    ----------
    raw_data : DemandGenChecks
        Aggregated demand-generation check data.
    width : int, optional
        Panel width in pixels.  Defaults to 250.
    height : int, optional
        Panel height in pixels.  Defaults to 250.

    Returns
    -------
    alt.LayerChart
        Layered Altair chart (survival curve lines + hover rule).
    """

    wtp_pct_greater = raw_data.wtp_pct_greater

    wtp_pct_greater = wtp_pct_greater / 100

    _wtp_survival_curves = alt.Chart(
        wtp_pct_greater.stack().rename("pct_wtp").reset_index().eval("tf = tf + 1"),
        width=width,
        height=height,
        title=alt.TitleParams("Willingness to Pay Survival Curves", anchor="middle", fontSize=14),
    )
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["price_point"], empty=False)
    when_near = alt.when(nearest)

    wtp_survival_curves = _wtp_survival_curves.mark_line().encode(
        x=alt.X("price_point:Q", title="Price Point", axis=alt.Axis(format="$.0f")),
        y=alt.Y("pct_wtp:Q", title="Percentage Willing to Pay", axis=alt.Axis(format=".0%")),
        color=alt.Color("tf:O", title="Time Frame", scale=alt.Scale(scheme="plasma", reverse=True)),
    )
    # Draw a rule at the location of the selection
    rules = (
        _wtp_survival_curves.transform_pivot("tf", value="pct_wtp", groupby=["price_point"])
        .mark_rule(color="gray")
        .encode(
            x="price_point:Q",
            opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
            tooltip=[alt.Tooltip("price_point", title="Price", format="$.2f")]
            + [
                alt.Tooltip(str(c), title=f"TF {c}", type="quantitative", format=".2%")
                for c in wtp_pct_greater.index + 1
            ],
        )
        .add_params(nearest)
    )
    wtp_survival_curves = wtp_survival_curves + rules
    return wtp_survival_curves


def _viz_wtp_statistics(raw_data: DemandGenChecks, height: int, width: int) -> alt.TopLevelMixin:
    """Build the Willingness-to-Pay Statistics panel.

    Displays mean WTP as a line with points, the interquartile range (25th–75th
    percentile) as an error bar, and the median/quartile values as shaped points,
    all coloured by timeframe.

    Parameters
    ----------
    raw_data : DemandGenChecks
        Aggregated demand-generation check data.
    height : int
        Panel height in pixels.
    width : int
        Panel width in pixels.

    Returns
    -------
    alt.LayerChart
        Layered Altair chart combining the error bar, mean line, and shaped
        marker points.
    """
    color = alt.Color("tf:O", title="Time Frame", scale=alt.Scale(scheme="plasma", reverse=True), legend=None)

    _base = alt.Chart(
        raw_data.wtp_summary.reset_index()
        .eval("label_mean = 'Mean'")
        .eval("tf = tf + 1")
        .eval("label_median = 'Median'")
        .eval("label_q75 = '75th Pctile'")
        .eval("label_q25 = '25th Pctile'"),
        title=alt.TitleParams("Willingness to Pay Statistics", anchor="middle", fontSize=14),
        width=width,
        height=height,
    )
    _shape_scale = alt.Scale(
        domain=["Mean", "Median", "75th Pctile", "25th Pctile"],
        range=["circle", "square", "triangle-up", "triangle-down"],  # Custom mapping
    )
    tooltips = [
        alt.Tooltip("tf:O", title="Timeframe"),
        alt.Tooltip("mean:Q", title="Mean WTP", format="$.2f"),
        alt.Tooltip("q25:Q", title="25th Pctile WTP", format="$.2f"),
        alt.Tooltip("q50:Q", title="Median WTP", format="$.2f"),
        alt.Tooltip("q75:Q", title="75th Pctile WTP", format="$.2f"),
    ]

    x_enc = alt.X("tf:O", title="Timeframe", axis=alt.Axis(labelAngle=0))

    return (
        # IQR error bar (Q25–Q75)
        _base.mark_errorbar(thickness=2, color="#ebebeb").encode(
            x=x_enc,
            y=alt.Y("q25:Q", title="Willingness to Pay"),
            y2=alt.Y2("q75:Q"),
            tooltip=tooltips,
        )
        # Mean line
        + _base.mark_line().encode(
            x=x_enc,
            y=alt.Y("mean:Q", title="Willingness to Pay"),
        )
        # Mean point
        + _base.mark_point(filled=True, opacity=1.0).encode(
            x=x_enc,
            y=alt.Y("mean:Q", title="Willingness to Pay"),
            size=alt.value(100),
            shape=alt.Shape("label_mean:N", legend=alt.Legend(title="WTP Stats"), scale=_shape_scale),
            color=color,
            tooltip=tooltips,
        )
        # Median point
        + _base.mark_point(filled=True, opacity=1.0).encode(
            x=x_enc,
            y=alt.Y("q50:Q", title="Willingness to Pay"),
            size=alt.value(100),
            shape=alt.Shape("label_median:N", legend=alt.Legend(title="WTP Stats"), scale=_shape_scale),
            color=color,
            tooltip=tooltips,
        )
        # 25th-percentile point
        + _base.mark_point(filled=True, opacity=1.0).encode(
            x=x_enc,
            y=alt.Y("q25:Q", title="Willingness to Pay"),
            size=alt.value(100),
            shape=alt.Shape("label_q25:N", legend=alt.Legend(title="WTP Stats"), scale=_shape_scale),
            color=color,
            tooltip=tooltips,
        )
        # 75th-percentile point
        + _base.mark_point(filled=True, opacity=1.0).encode(
            x=x_enc,
            y=alt.Y("q75:Q", title="Willingness to Pay"),
            size=alt.value(100),
            shape=alt.Shape("label_q75:N", legend=alt.Legend(title="WTP Stats"), scale=_shape_scale),
            color=color,
            tooltip=tooltips,
        )
    )


def _viz_volume(raw_data: DemandGenChecks, height: int, width: int) -> alt.Chart:
    """Build the Mean Segment Demand by Timeframe panel.

    Renders a stacked bar chart showing the simulated mean demand for each
    passenger segment across all booking timeframes.

    Parameters
    ----------
    raw_data : DemandGenChecks
        Aggregated demand-generation check data.
    height : int
        Panel height in pixels.
    width : int
        Panel width in pixels.

    Returns
    -------
    alt.Chart
        Altair bar chart with one stacked bar per timeframe, coloured by
        segment.
    """
    segments = raw_data.segments

    volume_data = (
        pd.DataFrame(
            raw_data.welford.mean,
            index=pd.Index(segments, name="segment"),
            columns=pd.Index(range(1, raw_data.welford.mean.shape[1] + 1), name="tf"),
        )
        .stack()
        .rename("mean_demand")
        .reset_index()
    )
    # Preserve the original segment order for consistent colour mapping
    volume_data["segment_order"] = volume_data["segment"].map({s: n for n, s in enumerate(segments)})

    return (
        alt.Chart(
            volume_data,
            width=width,
            height=height,
            title=alt.TitleParams("Mean Segment Demand by Timeframe", anchor="middle", fontSize=14),
        )
        .mark_bar()
        .encode(
            x=alt.X("tf:O", title="Time Frame", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("mean_demand:Q", title="Mean Demand"),
            order=alt.Order("segment_order:N"),
            color=alt.Color("segment:N", title="Segment", sort=segments),
            tooltip=[
                alt.Tooltip("segment:N", title="Segment"),
                alt.Tooltip("tf:O", title="Timeframe"),
                alt.Tooltip("mean_demand:Q", title="Mean Demand", format=".2f"),
            ],
        )
    )


def _viz_check_demand_generation(
    raw_data: DemandGenChecks,
    panel_widths: tuple[int, int, int, int, int] = (250, 250, 100, 350, 350),
    panel_height: int = 250,
) -> alt.VConcatChart:
    """Assemble the full five-panel demand-generation dashboard.

    Combines all individual visualisation panels into a two-row Altair layout:

    * **Row 1**: WTP survival curves | WTP statistics | Segment stats
    * **Row 2**: Segment volume by timeframe | Unconditional Frat5

    Each panel is built by its own ``_viz_*`` helper function so that panels
    can also be rendered independently.

    Parameters
    ----------
    raw_data : DemandGenChecks
        Aggregated demand-generation check data returned by
        :func:`check_demand_generation` with ``raw_data=True``.
    panel_widths : tuple of five ints, optional
        Pixel widths for each of the five panels in the order:
        (survival curves, WTP statistics, segment stats, Frat5, volume).
        Defaults to ``(250, 250, 100, 350, 350)``.
    panel_height : int, optional
        Shared pixel height applied to all panels.  Defaults to 250.

    Returns
    -------
    alt.VConcatChart
        A fully assembled Altair compound chart ready for display in a
        Jupyter notebook.
    """
    orig = raw_data.orig
    dest = raw_data.dest

    wtp_survival_curves = _viz_wtp_survival_curves(raw_data, width=panel_widths[0], height=panel_height)
    wtp_statistics = _viz_wtp_statistics(raw_data, height=panel_height, width=panel_widths[1])
    segment_stats_chart = _viz_segment_stats(raw_data, height=panel_height, width=panel_widths[2])
    frat_5_est = _viz_frat5(raw_data, height=panel_height, width=panel_widths[3])
    volume = _viz_volume(raw_data, height=panel_height, width=panel_widths[4])

    return (
        (
            (wtp_survival_curves | wtp_statistics | segment_stats_chart).resolve_scale(
                color="independent", shape="independent"
            )
            & (volume | frat_5_est).resolve_scale(color="independent", shape="independent")
        )
        .resolve_scale(color="independent", shape="independent")
        .properties(
            title=f"Demand and Willingness to Pay for {orig}~{dest} "
            f"(Min Reference Price: ${min(raw_data.reference_price.values()):,.0f})"
        )
    )


def check_mean_wtp_by_demand(cfg: Config, *, n_draws: int = 10_000):
    """Compute mean and standard-deviation of WTP for every demand config.

    Instantiates a lightweight set of core objects (booking curves, choice
    models, demands) from *cfg* and draws *n_draws* WTP samples for each
    demand.  The results are indexed by origin, destination, and segment.

    Parameters
    ----------
    cfg : Config
        Simulation configuration containing the demand and choice-model
        definitions to evaluate.
    n_draws : int, optional
        Number of random WTP draws per demand.  Defaults to 10 000.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a MultiIndex of ``(orig, dest, segment)`` and columns
        ``mean_wtp`` and ``std_wtp``.
    """
    fare_restriction_mapping = StringTracker(start_from=1, case_sensitive=False)
    prng = Generator(seed=42)

    booking_curves = {bc: make_core_booking_curve(cfg.booking_curves[bc], prng) for bc in cfg.booking_curves}
    choice_models = {
        cm: make_core_choice_model(cfg.choice_models[cm], prng, fare_restriction_mapping) for cm in cfg.choice_models
    }
    demands = [
        make_core_demand(
            dmdconfig,
            markets={},
            choice_models=choice_models,
            booking_curves=booking_curves,
        )
        for dmdconfig in cfg.demands
    ]

    out = pd.DataFrame({d.identifier: d.choice_model.max_wtp(d.reference_price, n_draws=n_draws) for d in demands}).T
    out = out.rename_axis(index="demand").rename(columns={"mean": "mean_wtp", "stdev": "std_wtp"})
    out.index = out.index.str.split(r"[~@]", expand=True).rename(["orig", "dest", "segment"])
    return out


def get_market_reference_prices(
    cfg: Config,
) -> dict[str, float]:
    """Get reference prices by market.

    Parameters
    ----------
    cfg : Config
        The configuration to scan.

    Raises
    ------
    ValueError
        If all demands that share a market do not also share a common reference price.
    """
    market_reference_prices = {}
    for d in cfg.demands:
        if d.market_identifier not in market_reference_prices:
            market_reference_prices[d.market_identifier] = d.reference_price
        else:
            if market_reference_prices[d.market_identifier] != d.reference_price:
                raise ValueError(f"inconsistent market reference prices in market {d.market_identifier}")
    return market_reference_prices


def check_reference_price_scaling(
    cfg: Config,
) -> pd.DataFrame:
    """Verify that reference prices scale consistently relative to minimum fare prices.

    For each market in *cfg*, computes the ratio of the configured reference
    price to the cheapest available fare price.  Raises an error if the ratio
    is not the same across all markets, which would indicate an inconsistent
    pricing setup.

    Parameters
    ----------
    cfg : Config
        Simulation configuration to validate.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per market and columns ``reference_price`` and
        ``min_price``.

    Raises
    ------
    ValueError
        If the reference-price / min-fare ratio is not consistent across all
        markets (checked with ``numpy.isclose``).
    """
    df = pd.concat(
        [
            pd.Series(get_market_reference_prices(cfg), name="reference_price"),
            pd.DataFrame.from_dict(check_min_fare_price_by_market(cfg), orient="index")["min_price"],
        ],
        axis=1,
    )
    ratio = df["reference_price"] / df["min_price"]
    if not np.isclose(ratio.min(), ratio.max()):
        raise ValueError(f"Ratio in in range {ratio.min()} to {ratio.max()}")
    logging.getLogger("passengersim.config.checks").info("Reference price scaling ratio is consistent: %f", ratio.min())
    return df
