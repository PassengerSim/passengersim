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

    @sim.end_sample_callback
    def collect_data2(sim):
        if sim.sim.sample % 5 == 0:
            data = {}
            for b in sim.legs[111].buckets:
                data[f"{b.name}-value"] = b.prorated_value
            return data

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
    assert summary.callback_data.end_sample[1] == approx(
        {
            "trial": 0,
            "sample": 5,
            "Y0-value": 500.47215380153233,
            "Y1-value": 408.1463017322425,
            "Y2-value": 300.4618895884555,
            "Y3-value": 221.87126664733074,
            "Y4-value": 170.345592374312,
            "Y5-value": 135.69208971756063,
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


@mark.parametrize("mp", [False, True])
def test_daily_callbacks(mp, dataframe_regression):
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))

    cfg.simulation_controls.num_samples = 100
    cfg.simulation_controls.burn_samples = 50
    cfg.simulation_controls.num_trials = 2
    cfg.db = None
    cfg.outputs.reports.clear()

    cfg.carriers.AL1.rm_system = "P"

    sim = pax.Simulation(cfg)

    @sim.end_sample_callback
    def collect_carrier_revenue(sim):
        if sim.sim.sample < sim.sim.burn_samples:
            return
        return {c.name: c.revenue for c in sim.sim.carriers}

    @sim.daily_callback
    def collect_carrier_revenue_detail(sim, days_prior):
        if sim.sim.trial == 1 or sim.sim.sample < sim.sim.burn_samples:
            return
        if sim.sim.sample % 7 == 0:
            return {c.name: c.revenue for c in sim.sim.carriers}

    summary = sim.run()

    assert summary.callback_data.daily[:5] == [
        {"trial": 0, "sample": 56, "days_prior": 63, "AL1": 0.0, "AL2": 0.0},
        {"trial": 0, "sample": 56, "days_prior": 62, "AL1": 600.0, "AL2": 2125.0},
        {"trial": 0, "sample": 56, "days_prior": 61, "AL1": 1350.0, "AL2": 4225.0},
        {"trial": 0, "sample": 56, "days_prior": 60, "AL1": 2025.0, "AL2": 6250.0},
        {"trial": 0, "sample": 56, "days_prior": 59, "AL1": 3125.0, "AL2": 6825.0},
    ]

    assert summary.callback_data.end_sample[:5] == [
        {"trial": 0, "sample": 50, "AL1": 100475.0, "AL2": 103700.0},
        {"trial": 0, "sample": 51, "AL1": 101475.0, "AL2": 97425.0},
        {"trial": 0, "sample": 52, "AL1": 108575.0, "AL2": 95000.0},
        {"trial": 0, "sample": 53, "AL1": 104275.0, "AL2": 98825.0},
        {"trial": 0, "sample": 54, "AL1": 101300.0, "AL2": 97000.0},
    ]

    df = summary.callback_data.to_dataframe("daily")
    dataframe_regression.check(df, "daily_callback")
