import logging
import warnings
from collections import namedtuple
from math import floor

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

from .choice_models import _interpolate_series_with_new_index

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
    ],
)


def _get_wtpQuantile_for_price(wtp_vector, price):
    """The fraction of passengers with maximum WTP greater than or equal to the given price."""
    return np.mean(wtp_vector >= price)


def _get_price_for_wtpQuantile(wtp_vector, quantile):
    """The price at which the given fraction of passengers have maximum WTP greater than or equal to that price."""
    return np.quantile(wtp_vector, 1 - quantile)


def _get_frat5_at_price(wtp_vector, starting_price):
    """Fare ratio at which half of passengers will disappear."""
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
    wtp_resolution: int = 200,
    raw_data: bool = False,
):
    """Check the random demand generation for a single market."""

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

    tf_boost_factor = (100_000 // (welford.mean * welford.n).astype(int).sum(0)) + 1
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
        _s = wtp_pct_greater.iloc[tf].rename("pct_wtp").reset_index().set_index("pct_wtp")
        _s = _s[~_s.index.duplicated(keep="last")]
        try:
            quantiles[tf] = _interpolate_series_with_new_index(
                _s["price_point"], pd.Index([5, 10, 12.5, 25, 37.5, 50, 75, 90, 99.9], name="pct_wtp")
            )
        except ValueError:
            print("FAIL")
            return _s

    quantiles_df = pd.concat(quantiles, axis=1, names=["tf"])
    quantiles_df.index = [f"q{floor(i):02d}" for i in quantiles_df.index]
    quantiles_df.index.name = "quantile"
    # quantiles_df.fillna(upper_bound, inplace=True)

    wtp_summary = pd.concat([wtp_summary, quantiles_df.T], axis=1)
    wtp_summary = wtp_summary.eval("frat5_99 = q50 / q99")
    wtp_summary = wtp_summary.eval("frat5_75 = q37 / q75")
    wtp_summary = wtp_summary.eval("frat5_50 = q25 / q50")
    wtp_summary = wtp_summary.eval("frat5_25 = q12 / q25")
    wtp_summary = wtp_summary.eval("frat5_10 = q05 / q10")

    segment_stats = pd.DataFrame(
        {s: {"mean": welford_by_segments[s].mean, "std": welford_by_segments[s].std_dev} for s in segments}
    ).T.rename_axis(index="segment")

    frat_5_estimates = {}
    fare_levels = {f.booking_class: f.price for f in cfg.fares}
    for bookclass, price in fare_levels.items():
        frat_5_estimates[(bookclass, price)] = [_get_frat5_at_price(wtp_vectors[tf], price) for tf in range(n_tf)]
    frat_5_estimates = pd.DataFrame.from_dict(frat_5_estimates).rename_axis(
        columns=["booking_class", "price"], index="tf"
    )

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
    )

    if raw_data:
        return output
    return _viz_check_demand_generation(output)


def _viz_frat5(raw_data: DemandGenChecks, height: int, width: int) -> alt.LayerChart:
    df = (
        pd.DataFrame.from_dict(raw_data.frat_5_estimates)
        .rename_axis(columns=["booking_class", "price"], index="tf")
        .T.stack()
        .rename("frat5value")
        .reset_index()
    )
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
        x=alt.X("tf:O", title="Timeframe"),
        y=alt.Y("frat5value", title="Frat5 Value", scale=alt.Scale(zero=False)),
        color=alt.Color("label:N", title="Booking Class"),
    )

    # Draw points on the line, and highlight based on selection
    _frat5points = _frat5lines.mark_point().encode(opacity=when_near.then(alt.value(1)).otherwise(alt.value(0)))

    _frat5rules = (
        alt.Chart(df)
        .transform_pivot("label", value="frat5value", groupby=["tf"])
        .mark_rule(color="gray")
        .encode(
            x=alt.X("tf:O"),
            opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
            tooltip=[alt.Tooltip("tf:O", title="Timeframe")]
            + [alt.Tooltip(c, type="quantitative", format=".3f") for c in df.label.unique()],
        )
        .add_params(nearest)
    )

    return alt.layer(_frat5lines, _frat5points, _frat5rules)


def _viz_segment_stats(raw_data: DemandGenChecks, height: int, width: int) -> alt.LayerChart:

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
) -> alt.VConcatChart:

    wtp_pct_greater = raw_data.wtp_pct_greater

    wtp_pct_greater = wtp_pct_greater / 100

    _wtp_survival_curves = alt.Chart(
        wtp_pct_greater.stack().rename("pct_wtp").reset_index(),
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
            + [alt.Tooltip(str(c), title=f"TF {c}", type="quantitative", format=".2%") for c in wtp_pct_greater.index],
        )
        .add_params(nearest)
    )
    wtp_survival_curves = wtp_survival_curves + rules
    return wtp_survival_curves


def _viz_check_demand_generation(
    raw_data: DemandGenChecks,
    panel_widths: tuple[int, int, int, int, int] = (250, 250, 100, 350, 350),
    panel_height: int = 250,
) -> alt.VConcatChart:

    orig = raw_data.orig
    dest = raw_data.dest
    wtp_summary = raw_data.wtp_summary
    welford = raw_data.welford
    segments = raw_data.segments

    wtp_survival_curves = _viz_wtp_survival_curves(
        raw_data,
        width=panel_widths[0],
        height=panel_height,
    )

    color = alt.Color("tf:O", title="Time Frame", scale=alt.Scale(scheme="plasma", reverse=True), legend=None)

    _wtp_statistics = alt.Chart(
        wtp_summary.reset_index()
        .eval("label_mean = 'Mean'")
        .eval("tf = tf + 1")
        .eval("label_median = 'Median'")
        .eval("label_q75 = '75th Pctile'")
        .eval("label_q25 = '25th Pctile'"),
        title=alt.TitleParams("Willingness to Pay Statistics", anchor="middle", fontSize=14),
        width=panel_widths[1],
        height=panel_height,
    )
    _shape_scale = alt.Scale(
        domain=["Mean", "Median", "75th Pctile", "25th Pctile"],
        range=["circle", "square", "triangle-up", "triangle-down"],  # Custom mapping
    )
    wtp_statistics_tooltips = [
        alt.Tooltip("tf:O", title="Timeframe"),
        alt.Tooltip("mean:Q", title="Mean WTP", format="$.2f"),
        alt.Tooltip("q25:Q", title="25th Pctile WTP", format="$.2f"),
        alt.Tooltip("q50:Q", title="Median WTP", format="$.2f"),
        alt.Tooltip("q75:Q", title="75th Pctile WTP", format="$.2f"),
    ]
    wtp_statistics = (
        _wtp_statistics.mark_errorbar(thickness=2, color="#ebebeb").encode(
            x=alt.X("tf:O", title="Timeframe", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("q25:Q", title="Willingness to Pay"),
            y2=alt.Y2("q75:Q", title="Willingness to Pay"),
            tooltip=wtp_statistics_tooltips,
        )
        + _wtp_statistics.mark_line().encode(
            x=alt.X("tf:O", title="Timeframe", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("mean:Q", title="Willingness to Pay"),
        )
        + _wtp_statistics.mark_point(filled=True, opacity=1.0).encode(
            x=alt.X("tf:O", title="Timeframe", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("mean:Q", title="Willingness to Pay"),
            size=alt.value(100),
            shape=alt.Shape("label_mean:N", legend=alt.Legend(title="WTP Stats"), scale=_shape_scale),
            color=color,
            tooltip=wtp_statistics_tooltips,
        )
        + _wtp_statistics.mark_point(filled=True, opacity=1.0).encode(
            x=alt.X("tf:O", title="Timeframe", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("q50:Q", title="Willingness to Pay"),
            size=alt.value(100),
            shape=alt.Shape("label_median:N", legend=alt.Legend(title="WTP Stats"), scale=_shape_scale),
            color=color,
            tooltip=wtp_statistics_tooltips,
        )
        + _wtp_statistics.mark_point(filled=True, opacity=1.0).encode(
            x=alt.X("tf:O", title="Timeframe", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("q25:Q", title="Willingness to Pay"),
            size=alt.value(100),
            shape=alt.Shape("label_q25:N", legend=alt.Legend(title="WTP Stats"), scale=_shape_scale),
            color=color,
            tooltip=wtp_statistics_tooltips,
        )
        + _wtp_statistics.mark_point(filled=True, opacity=1.0).encode(
            x=alt.X("tf:O", title="Timeframe", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("q75:Q", title="Willingness to Pay"),
            size=alt.value(100),
            shape=alt.Shape("label_q75:N", legend=alt.Legend(title="WTP Stats"), scale=_shape_scale),
            color=color,
            tooltip=wtp_statistics_tooltips,
        )
    )

    # frat_5_est = (
    #     alt.Chart(
    #         wtp_summary[["frat5_99", "frat5_75", "frat5_50", "frat5_25", "frat5_10"]]
    #         .rename_axis(columns="measure")
    #         .stack()
    #         .rename("frat5_value")
    #         .reset_index(),
    #         title=alt.TitleParams("Frat5 Estimates", anchor="middle", fontSize=14),
    #         width=panel_widths[3],
    #         height=panel_height,
    #     )
    #     .mark_line()
    #     .encode(
    #         x=alt.X("tf:O", title="Timeframe", axis=alt.Axis(labelAngle=0)),
    #         y=alt.Y("frat5_value:Q", title="Frat5 Values"),
    #         color=alt.Color("measure:N", title="Measure"),
    #     )
    # )
    frat_5_est = _viz_frat5(raw_data, height=panel_height, width=panel_widths[3])

    #### VOLUME

    volume_data = (
        pd.DataFrame(
            welford.mean,
            index=pd.Index(segments, name="segment"),
            columns=pd.Index(range(1, 17), name="tf"),
        )
        .stack()
        .rename("mean_demand")
        .reset_index()
    )
    volume_data["segment_order"] = volume_data["segment"].map({s: n for n, s in enumerate(segments)})

    volume = (
        alt.Chart(
            volume_data,
            width=panel_widths[4],
            height=panel_height,
            title=alt.TitleParams("Mean Segment Demand by Timeframe", anchor="middle", fontSize=14),
        )
        .mark_bar()
        .encode(
            x=alt.X("tf:O", title="Time Frame"),
            y=alt.Y(
                "mean_demand:Q",
                title="Mean Demand",
            ),
            order=alt.Order("segment_order:N"),
            color=alt.Color("segment:N", title="Segment", sort=segments),
            tooltip=[
                alt.Tooltip("segment:N", title="Segment"),
                alt.Tooltip("tf:O", title="Timeframe"),
                alt.Tooltip("mean_demand:Q", title="Mean Demand", format=".2f"),
            ],
        )
    )

    ####

    segment_stats_chart = _viz_segment_stats(raw_data, height=panel_height, width=panel_widths[2])

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
