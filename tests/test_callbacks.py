import pickle

from pytest import approx, mark

import passengersim as pax


# for the MP driver, the callback cannot be a local func inside the test,
# it needs to be defined at the module level
def collect_data(sim):
    if sim.sim.sample % 5 == 0:
        data = {}
        for b in sim.legs[101].buckets:
            data[f"{b.name}-value"] = b.prorated_value
        return data


@mark.parametrize("mp", [False, True])
def test_begin_sample_callback(mp):
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

    if mp:
        sim = pax.MultiSimulation(cfg)
    else:
        sim = pax.Simulation(cfg)

    sim.begin_sample_callback(collect_data)

    summary = sim.run()

    # check that callback was called on sim
    # this only happens on the single process driver, not MultiSimulation
    if not mp:
        assert isinstance(sim.callback_data, pax.callbacks.CallbackData)
        assert sim.callback_data.begin_sample[0] == {
            "trial": 0,
            "sample": 0,
            "Y0-value": 400.0,
            "Y1-value": 300.0,
            "Y2-value": 200.0,
            "Y3-value": 150.0,
            "Y4-value": 125.0,
            "Y5-value": 100.0,
        }

    # check that callback was added on summary
    # this should work on both single and multi process drivers
    assert isinstance(summary.callback_data, pax.callbacks.CallbackData)
    assert summary.callback_data.begin_sample[0] == {
        "trial": 0,
        "sample": 0,
        "Y0-value": 400.0,
        "Y1-value": 300.0,
        "Y2-value": 200.0,
        "Y3-value": 150.0,
        "Y4-value": 125.0,
        "Y5-value": 100.0,
    }
    assert summary.callback_data.begin_sample[1] == approx(
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

    pkl = pickle.dumps(summary)
    summary2 = pickle.loads(pkl)
    assert summary2.callback_data.begin_sample[0] == {
        "trial": 0,
        "sample": 0,
        "Y0-value": 400.0,
        "Y1-value": 300.0,
        "Y2-value": 200.0,
        "Y3-value": 150.0,
        "Y4-value": 125.0,
        "Y5-value": 100.0,
    }
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
