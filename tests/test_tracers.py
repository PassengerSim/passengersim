import numpy as np
from pytest import mark

import passengersim as pax
from passengersim.tracers.bid_price import LegBidPriceTracer, PathBidPriceTracer, fig_path_bid_prices
from passengersim.tracers.forecasts import (
    LegForecastTracer,
    PathForecastTracer,
    fig_path_forecast_dashboard,
)


def test_tracers():
    cfg = pax.Config.from_yaml([pax.demo_network("3mkt")])
    cfg.outputs._write_no_files()
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 50
    cfg.simulation_controls.burn_samples = 30
    cfg.simulation_controls.connection_builder.nonstop_leg_path_id_alignment = False

    pathids = [1, 2, 5, 6, 9, 10]  # All the AL1 paths

    tracer = PathBidPriceTracer(path_ids=pathids, priority=1)

    tracer2 = PathForecastTracer(path_ids=pathids)

    sim = pax.Simulation(cfg)
    sim.daily_callback(tracer)
    sim.begin_sample_callback(tracer2)
    summary = sim.run()

    fig_path_bid_prices(summary)

    fig_path_forecast_dashboard(summary, path_id=1)


@mark.parametrize("mp", [False, True])
def test_tracers_multitrial(mp: bool, dataframe_regression):
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 50
    cfg.simulation_controls.burn_samples = 30
    cfg.simulation_controls.connection_builder.nonstop_leg_path_id_alignment = False
    cfg.outputs._write_no_files()
    cfg.carriers.AL1.rm_system = "U"
    cfg.carriers.AL1.rm_system_options = {}
    cfg.carriers.AL1.frat5 = "curve_C"
    cfg.carriers.AL2.rm_system = "E"
    cfg.carriers.AL2.rm_system_options = {}
    if mp:
        sim = pax.MultiSimulation(cfg)
    else:
        sim = pax.Simulation(cfg)

    leg_ids = [101, 111, 201, 211]
    forecast_tracer = LegForecastTracer(leg_ids=leg_ids)
    forecast_tracer.attach(sim)

    path_ids = [1, 11]
    forecast_tracer2 = PathForecastTracer(path_ids=path_ids)
    forecast_tracer2.attach(sim)

    bp_leg_trace = LegBidPriceTracer(leg_ids=[101, 111])
    bp_leg_trace.attach(sim)

    bp_path_trace = PathBidPriceTracer(path_ids=[1, 5, 9])
    bp_path_trace.attach(sim)

    summary = sim.run()
    leg_fcst = summary.callback_data.selected_leg_forecasts
    pth_fcst = summary.callback_data.selected_path_forecasts

    # check that the combination tracer output is correctly scaled
    for leg_id in leg_ids:
        assert leg_fcst.query(f"leg_id == {leg_id}")[("history_closure", "Y5")].max() == 1

    for pth_id in path_ids:
        assert pth_fcst.query(f"path_id == {pth_id}")[("history_closure", "Y5")].max() == 1

    # the forecast part for legs does get made on AL2
    dataframe_regression.check(leg_fcst.astype(np.float32), basename="leg_forecasts_multitrial")

    # the forecast part for legs doesn't get made on AL1
    assert np.isnan(leg_fcst.loc[(101, 63), ("mean_to_departure", "Y5")])

    # the forecast part for paths does get made on AL1
    dataframe_regression.check(pth_fcst.astype(np.float32), basename="path_forecasts_multitrial")
    # the forecast part for paths doesn't get made on AL2
    assert np.isnan(pth_fcst.loc[(11, 63), ("mean_to_departure", "Y5")])

    # check the leg bid prices
    leg_bp = summary.callback_data.leg_bid_prices
    dataframe_regression.check(leg_bp.astype(np.float32), basename=f"leg_bid_prices_multitrial_mp{mp}")

    pth_bp = summary.callback_data.path_bid_prices
    dataframe_regression.check(pth_bp.astype(np.float32), basename=f"path_bid_prices_multitrial_mp{mp}")

    # TODO: std_dev on bid prices is different between mp True and False... why?
