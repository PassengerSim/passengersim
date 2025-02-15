import os
import pathlib
import re

import altair as alt

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

    # experiments.run should have written a report to the output directory
    assert isinstance(experiments.report_filename, pathlib.Path)
    filename1 = experiments.report_filename
    raw_report = experiments.report_filename.read_text()
    assert re.search(r"Figure 1: </span>Carrier Revenues</h2>", raw_report)

    # test the write_report method
    outfile = summaries.write_report("demo-output-1/meta.html", base_config=cfg)
    raw_output = outfile.read_text()
    assert re.search(r"Figure 1: </span>Carrier Revenues</h2>", raw_output)

    def nothing_to_see_here(summary):
        return (
            alt.Chart()
            .mark_text(
                text="these are not the droids you are looking for\nmove along",
                lineBreak="\n",
            )
            .properties(title="Jedi Mind Trick")
        )

    # test report after adding extra output sections
    experiments.extra_reporting = [
        "# Bonus Section",
        ("Bonus Figure", nothing_to_see_here),
        nothing_to_see_here,
    ]

    experiments.run()
    assert experiments.report_filename != filename1
    assert experiments.report_filename.exists()
    raw_report = experiments.report_filename.read_text()
    assert re.search(r"Figure 1: </span>Carrier Revenues</h2>", raw_report)
    assert re.search(r"<h1 id=\"bonus-section\">Bonus Section", raw_report)
    assert re.search(r"Figure [1-9]+: </span>Bonus Figure</h2>", raw_report)
    assert re.search(r"Figure [1-9]+: </span>Jedi Mind Trick</h2>", raw_report)
