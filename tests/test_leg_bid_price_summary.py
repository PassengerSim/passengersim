"""Public native bid-price summary views, persistence, and exports."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import passengersim as pax
from passengersim.summaries.generic import GenericSimulationTables
from passengersim.summaries.leg_bid_price_history import _aggregate_moments, _moments_from_rows


@pytest.fixture
def config() -> pax.Config:
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 4
    cfg.simulation_controls.burn_samples = 1
    cfg.simulation_controls.show_progress_bar = False
    cfg.simulation_controls.collect_leg_bid_prices = True
    cfg.outputs._write_no_files()
    cfg.outputs.reports.clear()
    cfg.db = None
    return cfg


def test_summary_columns_and_empty_cells() -> None:
    assert "n_observations" in pax.SimulationTables.leg_bid_price_history.__doc__
    raw = _moments_from_rows([(1, 2, 0, np.nan, 0.0), (1, 1, 1, 10.0, 0.0), (1, 0, 3, 20.0, 200.0)])
    summary = pax.SimulationTables(data={"leg_bid_price_history": raw})
    table = summary.leg_bid_price_history
    assert table.index.names == ["leg_id", "days_prior"]
    assert list(table) == ["bid_price_mean", "bid_price_std_dev", "n_observations"]
    assert table.n_observations.dtype == "int64"
    assert np.isnan(table.loc[(1, 2), "bid_price_mean"])
    assert np.isnan(table.loc[(1, 2), "bid_price_std_dev"])
    assert table.loc[(1, 1), "bid_price_mean"] == 10.0
    assert np.isnan(table.loc[(1, 1), "bid_price_std_dev"])
    assert table.loc[(1, 0), "bid_price_std_dev"] == pytest.approx(10.0)
    table.loc[(1, 0), "bid_price_mean"] = -1
    assert summary.leg_bid_price_history.loc[(1, 0), "bid_price_mean"] == 20.0


def test_summary_pooling_uses_counts_and_between_trial_variance() -> None:
    one = pax.SimulationTables(data={"leg_bid_price_history": _moments_from_rows([(1, 0, 1, 10.0, 0.0)])})
    three = pax.SimulationTables(data={"leg_bid_price_history": _moments_from_rows([(1, 0, 3, 30.0, 200.0)])})
    missing = pax.SimulationTables()
    merged = pax.SimulationTables(data={"leg_bid_price_history": _aggregate_moments([one, missing, three])})
    row = merged.leg_bid_price_history.loc[(1, 0)]
    assert row.n_observations == 4
    assert row.bid_price_mean == 25.0
    assert row.bid_price_std_dev == pytest.approx(np.std([10, 20, 30, 40], ddof=1))
    assert _aggregate_moments([missing]) is None


def test_custom_summary_payload_can_be_aggregated() -> None:
    moments = _moments_from_rows([(1, 0, 3, 20.0, 200.0)])
    custom = GenericSimulationTables(data={"leg_bid_price_history": moments})
    pd.testing.assert_frame_equal(_aggregate_moments([custom]), moments)


@pytest.mark.parametrize("mode", ["pxsim-lazy", "pxsim-eager", "pickle"])
def test_summary_round_trip_and_reaggregation(config: pax.Config, tmp_path: Path, mode: str) -> None:
    with pax.Simulation(config) as sim:
        summary = sim.run()
    if mode == "pickle":
        path = summary.to_pickle(tmp_path / "result.pkl", add_timestamp_ext=False)
        restored = pax.SimulationTables.from_pickle(path)
    else:
        path = summary.to_file(tmp_path / "result.pxsim", add_timestamp_ext=False)
        restored = pax.SimulationTables.from_file(path, lazy=mode == "pxsim-lazy")
    with restored:
        if mode == "pxsim-lazy":
            assert "leg_bid_price_history" not in restored._data
        # Aggregate before reading the public view to exercise lazy raw loading.
        combined = pax.SimulationTables.aggregate([restored, summary])
        actual = combined.leg_bid_price_history
        expected = summary.leg_bid_price_history
        pd.testing.assert_frame_equal(restored.leg_bid_price_history, expected)
        assert restored.db is None
        assert (actual.n_observations == 12).all()
        pd.testing.assert_series_equal(actual.bid_price_mean, expected.bid_price_mean)
        np.testing.assert_allclose(actual.bid_price_std_dev, expected.bid_price_std_dev * np.sqrt(10 / 11), rtol=1e-12)


def test_summary_single_observation_and_disabled(config: pax.Config) -> None:
    config.simulation_controls.num_trials = 1
    config.simulation_controls.num_samples = 2
    with pax.Simulation(config) as sim:
        table = sim.run().leg_bid_price_history
        assert (table.n_observations == 1).all()
        assert table.bid_price_std_dev.isna().all()
        sim.config.simulation_controls.collect_leg_bid_prices = False
        assert sim.run().leg_bid_price_history is None


def test_summary_excel_exports_public_columns(config: pax.Config, tmp_path: Path) -> None:
    with pax.Simulation(config) as sim:
        summary = sim.run(summarizer=pax.SimulationTables(items=["leg_bid_price_history"]))
    path = tmp_path / "result.xlsx"
    summary.to_xlsx(path)
    restored = pd.read_excel(path, sheet_name="leg_bid_price_history", index_col=[0, 1])
    pd.testing.assert_frame_equal(restored, summary.leg_bid_price_history, check_dtype=False)


def test_summary_cache_hit_preserves_table(config: pax.Config, tmp_path: Path) -> None:
    with pax.Simulation(config) as sim:
        first = sim.run(cache_dir=tmp_path)
        second = sim.run(cache_dir=tmp_path)
        pd.testing.assert_frame_equal(first.leg_bid_price_history, second.leg_bid_price_history)
        assert second.db is None
        second.close()
