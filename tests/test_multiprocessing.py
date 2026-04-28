from pytest import fixture, raises

import passengersim as pax


@fixture
def cfg() -> pax.Config:
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 40
    cfg.simulation_controls.burn_samples = 30
    return cfg


def test_multiprocessing_failure(cfg):
    cfg.carriers.AL1.rm_system = "V"
    cfg.carriers.AL1.frat5 = "curve_C"
    cfg.carriers.AL1.rm_system_options = dict(
        bid_price_vector=True,
    )
    sim = pax.MultiSimulation(cfg)
    with raises(
        ValueError, match="Keyword argument 'bid_price_vector' does not match any configuration parameter in 'V'."
    ):
        _summary = sim.run()
