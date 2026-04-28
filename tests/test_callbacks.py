import pickle

import pandas as pd
from pytest import mark

import passengersim as pax


# for the MP driver, the callback cannot be a local func inside the test,
# it needs to be defined at the module level
def collect_data(sim):
    if sim.eng.sample % 5 == 0:
        data = {}
        for b in sim.legs[101].buckets:
            data[f"{b.name}-value"] = b.prorated_value
        return data


@mark.parametrize("mp", [False, True])
def test_begin_sample_callback(mp, num_regression):
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))

    cfg.simulation_controls.num_samples = 100
    cfg.simulation_controls.burn_samples = 50
    cfg.simulation_controls.num_trials = 2
    cfg.db = None
    cfg.outputs.reports.clear()
    cfg.outputs._write_no_files()

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
        if sim.eng.sample % 5 == 0:
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
    num_regression.check(summary.callback_data.begin_sample[1], basename="begin_sample")
    num_regression.check(summary.callback_data.end_sample[1], basename="end_sample")

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
    num_regression.check(summary2.callback_data.begin_sample[1], basename="begin_sample")
    assert len(summary2.callback_data.begin_sample) == 40


@mark.parametrize("mp", [False, True])
def test_daily_callbacks(mp, dataframe_regression):
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))

    cfg.simulation_controls.num_samples = 100
    cfg.simulation_controls.burn_samples = 50
    cfg.simulation_controls.num_trials = 2
    cfg.db = None
    cfg.outputs.reports.clear()
    cfg.outputs._write_no_files()

    cfg.carriers.AL1.rm_system = "P"
    cfg.carriers.AL1.rm_system_options = {"bid_price_vector": False}
    sim = pax.Simulation(cfg)

    @sim.end_sample_callback
    def collect_carrier_revenue(sim):
        if sim.eng.sample < sim.eng.burn_samples:
            return
        return {c.name: c.revenue for c in sim.eng.carriers}

    @sim.daily_callback
    def collect_carrier_revenue_detail(sim, days_prior):
        if sim.eng.trial == 1 or sim.eng.sample < sim.eng.burn_samples:
            return
        if sim.eng.sample % 7 == 0:
            return {c.name: c.revenue for c in sim.eng.carriers}

    summary = sim.run()

    dataframe_regression.check(pd.DataFrame(summary.callback_data.daily), basename="daily_callback")
    assert summary.callback_data.daily[0] == {"trial": 0, "sample": 56, "days_prior": 63, "AL1": 0.0, "AL2": 0.0}

    dataframe_regression.check(pd.DataFrame(summary.callback_data.end_sample), basename="end_sample_callback")

    df = summary.callback_data.to_dataframe("daily")
    dataframe_regression.check(df, "daily_callback")
