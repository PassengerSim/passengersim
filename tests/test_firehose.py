import io

import passengersim as pax


def test_choice_firehose_filters_by_demand():
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 1
    cfg.simulation_controls.burn_samples = 0
    cfg.simulation_controls.show_progress_bar = False
    cfg.outputs.reports.clear()
    cfg.outputs._write_no_files()
    cfg.db = None

    sim = pax.Simulation(cfg)
    choice_log = io.StringIO()
    sim.attach_firehose(
        "choice",
        choice_log,
        orig="BOS",
        dest="ORD",
        segment="business",
    )

    sim.run()
    output = choice_log.getvalue()
    logged_demand_ids = {dmd.identifier for dmd in sim.demands if dmd.identifier in output}

    assert logged_demand_ids == {"BOS~ORD@business"}
