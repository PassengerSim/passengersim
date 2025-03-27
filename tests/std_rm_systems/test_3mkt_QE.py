import numpy as np

import passengersim as pax
from passengersim.config.manipulate import strip_all_restrictions
from passengersim.utils.regression_testing import deep_compare_obj


def test_simple_q_network():
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))
    strip_all_restrictions(cfg, inplace=True)

    cfg.carriers.AL1.rm_system = "Q"
    cfg.carriers.AL2.rm_system = "E"
    cfg.simulation_controls.num_trials = 1

    cfg.simulation_controls.num_samples = 10
    cfg.simulation_controls.burn_samples = 9
    cfg.simulation_controls.show_progress_bar = False

    sim = pax.Simulation(cfg)
    _summary = sim.run()

    pth = sim.paths.select(path_id=9)
    z = pth.q_forecast.history.as_mean_arrays()
    target_z = {
        "sold": np.asarray(
            [8.9, 3.8, 4.2, 4.5, 2.6, 2.3, 5.3, 4.2, 2.63031482, 2.96771246]
            + [4.7460282, 5.32181877, 3.77763855, 2.83322891, 4.38594989, 2.92396659]
        ),
        "sold_priceable": np.asarray(
            [8.9, 3.8, 4.2, 4.5, 2.6, 2.3, 5.3, 4.2, 2.63031482, 2.96771246]
            + [4.7460282, 5.32181877, 3.77763855, 2.83322891, 4.38594989, 2.92396659]
        ),
        "closure": np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 0.2, 0.2, 0.4, 0.5]
        ),
    }
    deep_compare_obj(z, target_z)

    y = pth.pathclasses.select(booking_class="Y0").history.as_mean_arrays()
    target_y = {
        "sold": np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        "sold_priceable": np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        "closure": np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.4, 0.5]
        ),
    }
    deep_compare_obj(y, target_y)
