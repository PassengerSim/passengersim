import numpy as np
from pytest import raises

import passengersim as pax
from passengersim.config.manipulate import strip_fare_restrictions
from passengersim.utils.regression_testing import deep_compare_obj


def test_simple_q_network_misconfigs():
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))
    strip_fare_restrictions(cfg, inplace=True)

    cfg.carriers.AL1.rm_system = "Q"
    cfg.carriers.AL2.rm_system = "E"
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 10
    cfg.simulation_controls.burn_samples = 9
    cfg.simulation_controls.show_progress_bar = False
    cfg.simulation_controls.connection_builder.nonstop_leg_path_id_alignment = False
    cfg.outputs._write_no_files()

    with raises(ValueError, match="Carrier 'AL1' must have a Frat5 curve defined for Q path forecasting"):
        _sim = pax.Simulation(cfg)

    cfg.carriers.AL1.frat5 = "curve_C"

    with raises(ValueError, match="Carrier 'AL1' must have 'store_q_history' set to True for Q path forecasting"):
        _sim = pax.Simulation(cfg)

    cfg.carriers.AL1.store_q_history = True
    _sim = pax.Simulation(cfg)

    cfg.frat5_curves["curve_C"].max_cap = 11.0
    with raises(
        ValueError,
        match="max_cap value 10.0 in QPathForecast does not match max_cap value 11.0 in Frat5 config for 'curve_C'.",
    ):
        _sim = pax.Simulation(cfg)


def test_simple_q_network():
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))
    strip_fare_restrictions(cfg, inplace=True)

    cfg.carriers.AL1.rm_system = "Q"
    cfg.carriers.AL2.rm_system = "E"
    # # use only old-style RM systems as there are some tiny differences in callback RM systems
    # for cxr in cfg.carriers.values():
    #     cxr.rm_system_options = False
    cfg.simulation_controls.num_trials = 1

    cfg.simulation_controls.num_samples = 10
    cfg.simulation_controls.burn_samples = 9
    cfg.simulation_controls.show_progress_bar = False
    cfg.simulation_controls.connection_builder.nonstop_leg_path_id_alignment = False
    cfg.outputs._write_no_files()

    cfg.carriers.AL1.frat5 = "curve_C"
    cfg.carriers.AL1.store_q_history = True

    sim = pax.Simulation(cfg)
    _summary = sim.run()

    pth = sim.paths.select(path_id=9)
    z = pth.q_forecast.history.as_mean_arrays()
    target_z = {
        "sold": np.asarray(
            [8.9, 3.9, 5.5, 3.6, 3.4, 1.8, 3.5, 3.3, 2.98899411, 2.51114131]
            + [4.3612151, 4.05471907, 2.20362249, 3.14803212, 7.51877123, 1.4619833]
        ),
        "sold_priceable": np.asarray(
            [8.9, 3.9, 5.5, 3.6, 3.4, 1.8, 3.5, 3.3, 2.98899411, 2.51114131]
            + [4.3612151, 4.05471907, 2.20362249, 3.14803212, 7.51877123, 1.4619833]
        ),
        "closure": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.6, 0.7]),
    }
    deep_compare_obj(z, target_z)

    y = pth.pathclasses.select(booking_class="Y0").history.as_mean_arrays()
    print(y)
    target_y = {
        "sold": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "sold_priceable": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "closure": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.6, 0.7]),
    }
    deep_compare_obj(y, target_y)
