"""Pooling and run isolation for native all-leg bid-price histories."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import passengersim as pax
from passengersim.experiments import Experiments
from passengersim.summaries.leg_bid_price_history import _merge_moments, _moments_from_rows


def make_config() -> pax.Config:
    """Build a small database-free run with native collection enabled."""
    config = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
    config.simulation_controls.num_trials = 2
    config.simulation_controls.num_samples = 4
    config.simulation_controls.burn_samples = 1
    config.simulation_controls.show_progress_bar = False
    config.simulation_controls.collect_leg_bid_prices = True
    config.outputs._write_no_files()
    config.outputs.reports.clear()
    config.db = None
    return config


def test_merge_unequal_counts_missing_cells_and_zero_observations() -> None:
    a = _moments_from_rows([(1, 0, 1, 10.0, 0.0), (2, 0, 0, np.nan, 0.0)])
    b = _moments_from_rows([(1, 0, 3, 30.0, 200.0), (3, 0, 1, 7.0, 0.0)])
    before_a, before_b = a.copy(), b.copy()
    result = _merge_moments(a, b)
    assert result.loc[(1, 0)].to_dict() == {"n_observations": 4, "mean": 25.0, "M2": 500.0}
    assert result.loc[(2, 0), "n_observations"] == 0
    assert np.isnan(result.loc[(2, 0), "mean"])
    assert result.loc[(2, 0), "M2"] == 0
    assert result.loc[(3, 0), "mean"] == 7.0
    pd.testing.assert_frame_equal(a, before_a)
    pd.testing.assert_frame_equal(b, before_b)
    pd.testing.assert_frame_equal(result, _merge_moments(b, a))


def test_uneven_partition_merges_match_raw_observations() -> None:
    """Different merge orders must preserve variance for large, similar prices."""
    values = 1e6 + np.random.default_rng(42).normal(size=103)
    partitions = np.split(values, [1, 8, 32])
    frames = [
        _moments_from_rows([(17, 0, len(part), part.mean(), ((part - part.mean()) ** 2).sum())]) for part in partitions
    ]
    for order in (frames, list(reversed(frames))):
        result = None
        for frame in order:
            result = _merge_moments(result, frame)
        row = result.loc[(17, 0)]
        assert row.n_observations == len(values)
        assert row["mean"] == pytest.approx(values.mean(), abs=1e-9)
        assert row.M2 / (row.n_observations - 1) == pytest.approx(values.var(ddof=1), rel=1e-9)


@pytest.mark.parametrize("execution", ["single", "sequential", "parallel"])
def test_pool_trials_and_workers(execution: str) -> None:
    config = make_config()
    with pax.Simulation(config) as reference:
        reference_summary = reference.run()
        expected = reference_summary._raw_leg_bid_price_history
        expected_table = reference_summary.leg_bid_price_history
    sim = pax.Simulation(config) if execution == "single" else pax.MultiSimulation(config)
    with sim:
        if execution == "sequential":
            summary = sim.sequential_run()
        elif execution == "parallel":
            summary = sim.run(max_processes=2)
        else:
            summary = sim.run()
        actual = summary._raw_leg_bid_price_history
        assert (actual.n_observations == 6).all()
        pd.testing.assert_frame_equal(actual, expected, rtol=1e-10, atol=1e-8)
        pd.testing.assert_frame_equal(summary.leg_bid_price_history, expected_table, rtol=1e-10, atol=1e-8)


def test_run_and_run_trial_clear_previous_observations() -> None:
    with pax.Simulation(make_config()) as sim:
        first = sim.run()._raw_leg_bid_price_history.copy()
        second = sim.run()._raw_leg_bid_price_history
        assert (first.n_observations == 6).all()
        assert (second.n_observations == 6).all()
        single = sim.run_trial(1)._raw_leg_bid_price_history
        assert (single.n_observations == 3).all()
        assert sim.eng._raw_leg_bid_price_statistics() == []
        # Earlier summaries own their data, including after disabling collection.
        sim.config.simulation_controls.collect_leg_bid_prices = False
        assert sim.run()._raw_leg_bid_price_history is None
        assert (first.n_observations == 6).all()


def test_saved_worker_moments_merge_lazily(tmp_path: Path) -> None:
    with pax.Simulation(make_config()) as sim:
        one = sim.run(single_trial=0)
        path = one.to_file(tmp_path / "one.pxsim", add_timestamp_ext=False)
    with pax.SimulationTables.from_file(path) as saved:
        merged = pax.SimulationTables.aggregate([saved, one])._raw_leg_bid_price_history
        assert (merged.n_observations == 6).all()
        pd.testing.assert_series_equal(merged["mean"], one._raw_leg_bid_price_history["mean"])


@pytest.mark.parametrize("single_process", [False, True])
def test_experiments_isolate_collection(tmp_path: Path, single_process: bool) -> None:
    experiments = Experiments(make_config(), output_dir=tmp_path)

    @experiments
    def baseline(config: pax.Config) -> pax.Config:
        return config

    @experiments
    def disabled(config: pax.Config) -> pax.Config:
        config.simulation_controls.collect_leg_bid_prices = False
        return config

    results = experiments.run(single_process=single_process, use_existing=False, write_report=False)
    assert (results["baseline"]._raw_leg_bid_price_history.n_observations == 6).all()
    assert results["disabled"]._raw_leg_bid_price_history is None
