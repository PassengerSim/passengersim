"""Run isolated legacy/chunked storage measurements without timing assertions.

Example: python benchmark_pxsim_storage.py --output .ai/pxsim-storage/benchmark.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import resource
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import lz4
import numpy as np
import pandas as pd

from passengersim.utils.kvstore import KVStore


def data_for(case: str) -> dict[str, Any]:
    """Use deterministic, representative numeric and Python-object shapes."""
    rng = np.random.default_rng(13)
    if case == "numeric":
        return {"frame": pd.DataFrame(rng.random((250000, 8)))}
    if case == "mixed":
        return {
            "frame": pd.DataFrame(
                {
                    "trial": np.repeat(np.arange(25), 10000),
                    "carrier": pd.Categorical(np.tile(["AL1", "AL2"], 125000)),
                    "market": np.tile(["AAA-BBB", "BBB-CCC"], 125000),
                    "revenue": rng.random(250000),
                }
            )
        }
    if case == "callbacks":
        return {"callbacks": {i: {"sold": i % 200, "values": [i * 0.25, i * 0.5]} for i in range(50000)}}
    return {str(i): pd.DataFrame(rng.random((50, 8))) for i in range(100)}


def worker(case: str, version: int) -> dict[str, float | int]:
    """Measure complete archive operations in a fresh process."""
    with tempfile.TemporaryDirectory(prefix="pxsim-benchmark-") as directory:
        root = Path(directory)
        path = root / "result.pxsim"
        values = data_for(case)
        t = time.perf_counter()
        store = KVStore(path, format_version=version)
        with store.transaction() if version == 2 else contextlib.nullcontext():
            for key, value in values.items():
                store[key] = value
        store.close()
        write = time.perf_counter() - t
        write_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        store = KVStore(path, mode="r")
        t = time.perf_counter()
        first = store[next(iter(values))]
        first_read = time.perf_counter() - t
        del first
        t = time.perf_counter()
        restored = {key: store[key] for key in store}
        read = time.perf_counter() - t
        for key, value in values.items():
            if isinstance(value, pd.DataFrame):
                pd.testing.assert_frame_equal(value, restored[key])
            else:
                assert value == restored[key]
        del restored
        t = time.perf_counter()
        dest = KVStore(root / "copy.pxsim", format_version=version)
        with dest.transaction() if version == 2 else contextlib.nullcontext():
            for key in store:
                dest.copy_entry_from(store, key)
        dest.close()
        resave = time.perf_counter() - t
        store.close()
        return {
            "write_s": write,
            "first_read_s": first_read,
            "all_read_s": read,
            "resave_s": resave,
            "file_bytes": path.stat().st_size,
            "write_peak_rss": write_peak,
            "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }


def main() -> None:
    """Run isolated benchmark workers and write the collected measurements."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["numeric", "mixed", "callbacks", "small"])
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.version)))
        return
    output = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
            "pandas": pd.__version__,
            "lz4": lz4.__version__,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "rss_unit": "bytes on macOS; KiB on Linux",
            "cache": "warm OS cache, separate processes",
        },
        "cases": {},
    }
    for case in ["numeric", "mixed", "callbacks", "small"]:
        output["cases"][case] = {}
        for version in (1, 2):
            runs = []
            for repeat in range(6):
                result = subprocess.check_output(
                    [sys.executable, __file__, "--worker", case, "--version", str(version)], text=True
                )
                if repeat:
                    runs.append(json.loads(result))
            output["cases"][case][str(version)] = {
                "runs": runs,
                "median": {key: statistics.median(r[key] for r in runs) for key in runs[0]},
            }
        print(case, output["cases"][case]["2"]["median"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
