import pathlib
import re
from contextlib import chdir

import altair as alt
from pytest import raises, warns

import passengersim as pax
from passengersim.contrast import Contrast
from passengersim.experiments import Experiments, OverwriteExperimentWarning


def test_experiments_interface(tmp_path):
    with chdir(tmp_path):
        cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))
        cfg.carriers.AL1.rm_system = "E"
        cfg.carriers.AL2.rm_system = "U"  # so bid prices and displacement costs exist
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
        assert all(isinstance(summary, pax.SimulationTables) for summary in summaries.values())

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
        assert re.search(r"Figure [1-9]+: </span>Bid Price History</h2>", raw_report)

        raw_report_1_file = summaries["baseline"].config.outputs.get_output_filename("html")
        pattern = raw_report_1_file.with_suffix(".*.html")
        print("pattern\n", pattern.name)
        raw_report_1_files = list(pathlib.Path("demo-output-1/baseline").glob(pattern.name))
        if len(raw_report_1_files) >= 1:
            raw_report_1_file = sorted(raw_report_1_files)[-1]
        else:
            for i in [str(file) for file in pathlib.Path("demo-output-1").rglob("*") if file.is_file()]:
                print("found file in demo-output-1:", i)
            raise FileNotFoundError(raw_report_1_file)
        raw_report_1 = pathlib.Path(raw_report_1_file).read_text()
        assert re.search(r"Figure 1: </span>Carrier Revenues</h2>", raw_report_1)
        assert not re.search(r"<h1 id=\"bonus-section\">Bonus Section", raw_report_1)
        assert not re.search(r"Figure [1-9]+: </span>Bonus Figure</h2>", raw_report_1)
        assert not re.search(r"Figure [1-9]+: </span>Jedi Mind Trick</h2>", raw_report_1)
        assert re.search(r"Figure [1-9]+: </span>Bid Price History</h2>", raw_report_1)
        assert re.search(r"Figure [1-9]+: </span>Displacement Cost History</h2>", raw_report_1)


def test_experiments_rerun(tmp_path):
    with chdir(tmp_path):
        cfg = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
        cfg.carriers.AL1.rm_system = "E"
        cfg.carriers.AL2.rm_system = "E"
        cfg.carriers.AL1.rm_system_options = {}
        cfg.carriers.AL2.rm_system_options = {}
        cfg.simulation_controls.num_trials = 1
        cfg.simulation_controls.num_samples = 15
        cfg.simulation_controls.burn_samples = 13
        experiments = Experiments(cfg, output_dir="demo-output-2")

        @experiments
        def low_dmd(cfg: pax.Config) -> pax.Config:
            cfg.simulation_controls.demand_multiplier = 0.9
            return cfg

        # no existing saved result, check "raise" for use_existing
        with raises(FileNotFoundError, match="demo-output-2/low_dmd/.*.pxsim"):
            _ = experiments.run(use_existing="raise")

        # no existing saved result, if "ignore" for use_existing should get empty
        summary_0 = experiments.run(use_existing="ignore", write_report=False)
        assert len(summary_0) == 0

        # our first run should actually run the experiment.
        summary_1 = experiments.run()
        assert "low_dmd" in summary_1
        assert isinstance(summary_1["low_dmd"], pax.summaries.SimulationTables)

        # our second run should not need to rerun anything, it's got cached results
        # so calling with "raise" not not raise any error
        summary_2 = experiments.run(use_existing="raise")
        assert "low_dmd" in summary_2
        assert isinstance(summary_2["low_dmd"], pax.summaries.SimulationTables)

        with warns(OverwriteExperimentWarning):
            # now change the experiment
            @experiments
            def low_dmd(cfg: pax.Config) -> pax.Config:
                cfg.simulation_controls.demand_multiplier = 0.95
                return cfg

        # existing saved result does not match experiment config, should complain
        with raises(ValueError, match="existing result does not match requested experiment"):
            _ = experiments.run(use_existing="raise")

        # if we don't check content, the discrepancy should be plowed over
        summary_3 = experiments.run(use_existing="raise", check_content=False)
        assert "low_dmd" in summary_3
        assert isinstance(summary_3["low_dmd"], pax.summaries.SimulationTables)

        # if the cache is missing, but the file is on disk, it should be able to use that
        experiments.experiments[0].cached = None
        summary_4 = experiments.run(use_existing="raise", check_content=False)
        assert "low_dmd" in summary_4
        assert isinstance(summary_4["low_dmd"], pax.summaries.SimulationTables)
        assert "available in file storage" in repr(summary_4["low_dmd"])

        # if the disk file is missing, and also the cache is missing, we should get an error
        experiments.experiments[0].cached = None
        for pxfile in experiments.output_dir.joinpath("low_dmd").rglob("*.pxsim"):
            pxfile.unlink()
        with raises(FileNotFoundError, match="demo-output-2/low_dmd/.*.pxsim"):
            _ = experiments.run(use_existing="raise", check_content=False)
