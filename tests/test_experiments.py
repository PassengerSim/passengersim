import os
import re

import passengersim as pax
from passengersim.contrast import Contrast
from passengersim.experiments import Experiments


def test_experiments_interface(tmp_path):
    os.chdir(tmp_path)
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))
    cfg.carriers.AL1.rm_system = "E"
    cfg.carriers.AL2.rm_system = "E"
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 50
    cfg.simulation_controls.burn_samples = 30
    cfg.db = None
    cfg.outputs.reports.clear()
    cfg.outputs.html.filename = "summary.html"
    experiments = Experiments(cfg, output_dir="demo-output-1")

    @experiments
    def baseline(cfg: pax.Config) -> pax.Config:
        return cfg

    @experiments
    def low_dmd(cfg: pax.Config) -> pax.Config:
        cfg.simulation_controls.demand_multiplier = 0.9
        return cfg

    summaries = experiments.run()
    assert isinstance(summaries, Contrast)
    assert summaries.keys() == {"baseline", "low_dmd"}
    assert all(
        isinstance(summary, pax.SimulationTables) for summary in summaries.values()
    )
    outfile = summaries.write_report("demo-output-1/meta.html", base_config=cfg)
    raw_output = outfile.read_text()
    assert re.search(r"Figure 1: </span>Carrier Revenues</h2>", raw_output)
