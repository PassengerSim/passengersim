import pandas as pd
from pytest import approx, fixture, mark

import passengersim as pax


# for the MP driver, the callback cannot be a local func inside the test,
# it needs to be defined at the module level
def collect_data(sim):
    if sim.sim.sample % 5 == 0:
        data = {}
        for b in sim.legs[101].buckets:
            data[f"{b.name}-value"] = b.prorated_value
        return data


@fixture(scope="module")
def tmp_path_module(tmp_path_factory):
    return tmp_path_factory.mktemp("test-io")


@fixture(scope="module")
def summary(tmp_path_module):
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))

    cfg.simulation_controls.num_samples = 100
    cfg.simulation_controls.burn_samples = 50
    cfg.simulation_controls.num_trials = 2
    cfg.db = None
    cfg.outputs.reports.clear()

    cfg.choice_models.business.restrictions["toxic"] = 99.99
    cfg.choice_models.leisure.restrictions["toxic"] = 99.99
    for f in cfg.fares:
        if f.booking_class == "Y5":
            f.restrictions.append("toxic")

    sim = pax.Simulation(cfg)

    sim.begin_sample_callback(collect_data)

    summary = sim.run()
    summary.to_file(tmp_path_module / "summary-io")
    return summary


@mark.parametrize("lazy", [False, True])
def test_summary_file(summary, tmp_path_module, lazy):
    summary2 = pax.SimulationTables.from_file(tmp_path_module / "summary-io", lazy=lazy)

    assert "summary-io." in summary2.metadata("store.filename")
    assert summary2.metadata("store.filename").endswith(".pxsim")

    # check that callback was added on summary
    assert isinstance(summary2.callback_data, pax.callbacks.CallbackData)
    assert summary2.callback_data.begin_sample[1] == approx(
        {
            "trial": 0,
            "sample": 5,
            "Y0-value": 282.41525649702044,
            "Y1-value": 262.9398718320564,
            "Y2-value": 178.64916602008478,
            "Y3-value": 135.9406111272244,
            "Y4-value": 97.39804088978694,
            "Y5-value": 74.11684420434234,
        }
    )
    assert len(summary2.callback_data.begin_sample) == 40

    # check that some summary is the same
    pd.testing.assert_frame_equal(summary.carriers, summary2.carriers)
    pd.testing.assert_frame_equal(summary.legs, summary2.legs)
