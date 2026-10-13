"""Archives and disk caches retain summaries independently of detail databases."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import passengersim as pax
from passengersim.config.database import DatabaseConfig
from passengersim.database import Database
from passengersim.experiments import Experiments
from passengersim.summaries.generic import GenericSimulationTables
from passengersim.utils.kvstore import KVStore


@pytest.fixture
def config() -> pax.Config:
    """Use a small real simulation with no automatic report files."""
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 3
    cfg.simulation_controls.burn_samples = 1
    cfg.simulation_controls.show_progress_bar = False
    cfg.db = DatabaseConfig()
    cfg.outputs._write_no_files()
    cfg.outputs.reports.clear()
    return cfg


def test_archive_does_not_touch_detail_database(tmp_path: Path) -> None:
    """Even pending borrowed changes stay outside the summary archive."""
    detail = tmp_path / "details.sqlite"
    connection = sqlite3.connect(detail)
    database = Database(connection=connection)
    connection.execute("CREATE TABLE details(value BLOB)")
    connection.execute("INSERT INTO details VALUES (zeroblob(1000000))")
    frame = pd.DataFrame({"value": [42]})
    result = GenericSimulationTables(db=database, data={"report": frame}, n_total_samples=1)
    archive = result.to_file(tmp_path / "summary.pxsim", add_timestamp_ext=False)
    assert connection.in_transaction  # Export must not commit the user's data.
    with sqlite3.connect(archive) as check:
        assert {r[0] for r in check.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {
            "pxsim_format",
            "pxsim_entries",
            "pxsim_payloads",
            "pxsim_chunks",
        }
        assert check.execute("SELECT DISTINCT serializer FROM pxsim_payloads").fetchall() == [("pickle",)]
    result.close()
    connection.close()
    detail.unlink()
    with GenericSimulationTables.from_file(archive, lazy=False) as loaded:
        pd.testing.assert_frame_equal(loaded._data["report"], frame)
        assert not loaded.database_available
        assert loaded.database_path is None
    store = KVStore(archive, mode="r")
    store.validate(full=True)
    store.close()


@pytest.mark.parametrize("kind", [pax.Simulation, pax.MultiSimulation])
def test_cache_hit_after_detail_deletion(
    config: pax.Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: type
) -> None:
    """A deleted detail database must not invalidate cached summaries or cause a run."""
    config.db.filename = tmp_path / "details.sqlite"
    first_sim = kind(config)
    first = first_sim.run(cache_dir=tmp_path / "cache")
    expected = first.carriers.copy()
    assert first.database_available
    first.close()
    first_sim.close()
    config.db.filename.unlink()

    def unexpected_run(*args: object, **kwargs: object) -> None:
        pytest.fail("A summary-only cache hit must not rerun the simulation")

    from passengersim.driver import _cache_run

    monkeypatch.setattr(kind, "run", unexpected_run)
    with kind(config) as sim:
        loaded = _cache_run.cache_run(sim, tmp_path / "cache")
        pd.testing.assert_frame_equal(loaded.carriers, expected)
        assert not loaded.database_available
        assert not sim.database_available
        assert not config.db.filename.exists()
        loaded.close()


@pytest.mark.parametrize("single_process", [True, False])
def test_experiment_disk_cache_is_summary_only(
    config: pax.Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, single_process: bool
) -> None:
    """Experiment disk results remain reusable after their detail files are removed."""
    config.outputs.disk = True
    config.db.result_storage = "persistent"
    config.db.result_directory = tmp_path / "details"
    experiments = Experiments(config, output_dir=tmp_path / "outputs")

    @experiments
    def baseline(cfg: pax.Config) -> pax.Config:
        return cfg

    first = experiments.run(single_process=single_process, cache_results=False, write_report=False)
    expected = first["baseline"].carriers.copy()
    path = first["baseline"].database_path
    first["baseline"].close()
    path.unlink()

    def unexpected_run(*args: object, **kwargs: object) -> None:
        pytest.fail("An experiment with saved summaries must not run again")

    monkeypatch.setattr(pax.MultiSimulation, "sequential_run", unexpected_run)
    monkeypatch.setattr(pax.MultiSimulation, "_run_asynchronously", unexpected_run)
    second = experiments.run(
        single_process=single_process, cache_results=False, write_report=False, use_existing="raise"
    )
    pd.testing.assert_frame_equal(second["baseline"].carriers, expected)
    assert not second["baseline"].database_available
    second["baseline"].close()
