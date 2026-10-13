"""Run with either checkout's interpreter to compare native-write throughput."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter

import passengersim as pax


def main():
    durations = []
    rows = []
    for _ in range(3):
        with tempfile.TemporaryDirectory() as temporary:
            cfg = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
            cfg.simulation_controls.num_trials = 1
            cfg.simulation_controls.num_samples = 100
            cfg.simulation_controls.burn_samples = 10
            cfg.simulation_controls.show_progress_bar = False
            cfg.db.filename = Path(temporary) / "benchmark.sqlite"
            cfg.db.write_items = {"leg", "bucket", "fare", "demand", "pathclass", "bookings"}
            cfg.outputs.disk = False
            cfg.outputs.html.filename = False
            cfg.outputs.pickle = False
            sim = pax.Simulation(cfg)
            start = perf_counter()
            summary = sim.run()
            durations.append(perf_counter() - start)
            with summary.db:
                rows.append(summary.db.execute("SELECT count(*) FROM leg_detail").fetchone()[0])
            summary.db.close()
    print(json.dumps({"seconds": durations, "median_seconds": median(durations), "leg_rows": rows}))


if __name__ == "__main__":
    main()
