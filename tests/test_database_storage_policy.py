"""Storage intent has the same meaning across all execution interfaces."""

import gc
import sqlite3
from pathlib import Path

import pytest

import passengersim as pax
from passengersim.config.database import DatabaseConfig
from passengersim.config.storage import resolve_storage
from passengersim.experiments import Experiments

pytestmark = pytest.mark.filterwarnings("error:.*cnx.*:DeprecationWarning")


@pytest.fixture
def config():
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 3
    cfg.simulation_controls.burn_samples = 1
    cfg.simulation_controls.show_progress_bar = False
    cfg.outputs._write_no_files()
    cfg.outputs.reports.clear()
    cfg.db = DatabaseConfig()
    return cfg


def run_model(config, execution):
    """Exercise the public entry points, including actual spawned workers."""
    sim = pax.Simulation(config) if execution == "single" else pax.MultiSimulation(config)
    if execution == "sequential":
        return sim, sim.sequential_run()
    if execution == "parallel":
        return sim, sim.run(max_processes=2)
    return sim, sim.run()


@pytest.mark.parametrize("execution", ["single", "sequential", "parallel"])
@pytest.mark.parametrize("mode", ["default", "temporary", "persistent", "summary_only", "disabled", "none"])
def test_execution_modes(config, tmp_path, execution, mode):
    if mode == "none":
        config.db = None
    elif mode == "disabled":
        config.db.engine = None
    elif mode != "default":
        config.db.result_storage = mode
    if mode in {"temporary", "persistent"}:
        config.db.result_directory = tmp_path / "results"
    if mode == "summary_only":
        config.outputs.reports = {"segmentation_detail"}
    sim, result = run_model(config, execution)
    if mode in {"none", "disabled", "summary_only"}:
        assert result.db is None
        assert not sim.database_available and not result.database_available
        assert sim.database_path is None and result.database_path is None
        if mode == "summary_only":
            assert not result.segmentation_detail.empty
    else:
        assert sim.database_available and result.database_available
        assert result.database_status["complete"]
        assert result.database_status["completed"] == [0, 1]
        path = result.database_path
        assert isinstance(path, Path) and path.is_absolute() and path.is_file()
        assert path == sim.database_path == Path(result.db.filename)
        assert result.db.execute("SELECT count(DISTINCT trial) FROM leg_detail").fetchone() == (2,)
        result.close()
        assert not sim.database_available and not result.database_available
        assert sim.database_path == result.database_path == path
        assert path.exists() == (mode == "persistent")
    sim.close()


@pytest.mark.parametrize("alias", [False, True])
def test_real_memory_and_export(config, tmp_path, alias):
    if alias:
        config.db.filename = ":memory:"
    else:
        config.db.result_storage = "memory"
    sim, result = run_model(config, "single")
    assert result.database_available and result.database_path is None
    assert result.db.execute("PRAGMA database_list").fetchone()[2] == ""
    saved = result.to_file(tmp_path / "memory.pxsim", add_timestamp_ext=False)
    result.close()
    restored = pax.SimulationTables.from_file(saved)
    assert restored.database_path is None
    assert not restored.database_available
    assert restored.n_total_samples == result.n_total_samples
    restored.close()


@pytest.mark.parametrize("execution", ["sequential", "parallel"])
def test_memory_rejected_before_allocation(config, execution, tmp_path):
    config.db.result_storage = "memory"
    with pytest.raises(ValueError, match="standalone-only"):
        run_model(config, execution)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("execution", ["single", "sequential", "parallel"])
def test_exact_persistent_destination_is_not_overwritten(config, tmp_path, execution):
    path = tmp_path / "exact.sqlite"
    config.db.filename = path
    sim, result = run_model(config, execution)
    assert result.database_path == path
    result.close()
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        run_model(config, execution)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "settings",
    [
        {"result_storage": "temporary", "filename": "db.sqlite"},
        {"result_storage": "persistent"},
        {"filename": "db.sqlite", "result_directory": "scratch"},
        {"result_storage": "memory", "result_directory": "scratch"},
        {"result_storage": "summary_only", "filename": "db.sqlite"},
        {"result_storage": "temporary", "filename": ":memory:"},
    ],
)
def test_invalid_policy_is_rejected_without_creating_files(config, tmp_path, settings, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config.db = DatabaseConfig(**settings)
    with pytest.raises(ValueError):
        pax.Simulation(config)
    with pytest.raises(ValueError):
        pax.MultiSimulation(config).sequential_run()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("none", [False, True])
def test_disabled_database_reports_fail_early(config, tmp_path, none):
    if none:
        config.db = None
    else:
        config.db.engine = None
        config.db.result_directory = tmp_path / "unused"
    config.outputs.reports = {"segmentation_detail"}
    for execution in ("single", "sequential", "parallel"):
        with pytest.raises(ValueError, match="disabled"):
            run_model(config, execution)
    assert list(tmp_path.iterdir()) == []


def test_borrowed_connection_and_conflicting_settings(config, tmp_path):
    connection = sqlite3.connect(tmp_path / "user.sqlite")
    sim = pax.Simulation(config, connection=connection)
    result = sim.run()
    assert result.db.connection is connection
    assert result.database_path == tmp_path / "user.sqlite"
    result.close()
    assert connection.execute("SELECT 1").fetchone() == (1,)
    for settings in ({"result_storage": "temporary"}, {"filename": ":memory:"}, {"engine": None}):
        config.db = DatabaseConfig(**settings)
        with pytest.raises(ValueError, match="connection"):
            pax.Simulation(config, connection=connection)
    config.db = None
    with pytest.raises(ValueError, match="connection"):
        pax.Simulation(config, connection=connection)
    connection.rollback()
    connection.close()


def test_default_and_disabled_json_round_trips(config):
    assert config.db.filename is None
    for db in (DatabaseConfig(), DatabaseConfig(engine=None), None):
        config.db = db
        restored = pax.Config.model_validate_json(config.model_dump_json())
        assert resolve_storage(restored.db) == resolve_storage(db)
    config.db = DatabaseConfig(engine=None, result_storage="persistent", filename="later.sqlite")
    restored = pax.Config.model_validate_json(config.model_dump_json())
    restored.db.engine = "sqlite"
    assert resolve_storage(restored.db).mode == "persistent"


def test_single_rerun_and_raw_handle_ownership(config, tmp_path):
    config.db.result_directory = tmp_path
    sim, first = run_model(config, "single")
    first_path = first.database_path
    second = sim.run()
    assert first_path != second.database_path
    raw = first.db.connection
    del first
    gc.collect()
    assert first_path.exists() and raw.execute("SELECT 1").fetchone() == (1,)
    raw.close()
    assert not first_path.exists()
    assert second.database_available
    sim.close()


@pytest.mark.parametrize("single_process", [True, False])
@pytest.mark.parametrize("mode", ["temporary", "persistent", "summary_only", "disabled"])
def test_experiment_storage_modes(config, tmp_path, single_process, mode):
    if mode == "disabled":
        config.db = None
    else:
        config.db.result_storage = mode
        if mode != "summary_only":
            config.db.result_directory = tmp_path / "data"
    experiments = Experiments(config, output_dir=tmp_path / "outputs")

    @experiments
    def baseline(cfg):
        return cfg

    @experiments
    def alternative(cfg):
        return cfg

    results = experiments.run(single_process=single_process, retain_sims=True, use_existing=False, write_report=False)
    paths = []
    for tag, result in results.items():
        assert result.database_available == (mode in {"temporary", "persistent"})
        assert result.database_path == experiments.sims[tag].database_path
        if result.database_available:
            paths.append(result.database_path)
            assert tag in result.database_path.parts
        result.close()
    assert len(paths) == len(set(paths))
    assert all(path.exists() == (mode == "persistent") for path in paths)


@pytest.mark.parametrize("single_process", [True, False])
def test_experiments_reject_memory_and_duplicate_destinations(config, tmp_path, single_process):
    experiments = Experiments(config, output_dir=tmp_path / "outputs")

    @experiments
    def baseline(cfg):
        return cfg

    @experiments
    def alternative(cfg):
        return cfg

    config.db.result_storage = "memory"
    with pytest.raises(ValueError, match="standalone-only"):
        experiments.run(single_process=single_process, use_existing=False, write_report=False)
    config.db.result_storage = "persistent"
    config.db.filename = tmp_path / "conflict.sqlite"
    with pytest.raises(ValueError, match="share"):
        experiments.run(single_process=single_process, use_existing=False, write_report=False)
    assert not config.db.filename.exists()


def test_single_cache_hit_and_rerun_keep_old_results(config, tmp_path):
    first = pax.Simulation(config).run(cache_dir=tmp_path / "cache")
    first.close()
    sim = pax.Simulation(config)
    loaded = sim.run(cache_dir=tmp_path / "cache")
    loaded_path = loaded.database_path
    assert not loaded.database_available and loaded.database_path is None
    fresh = sim.run()
    assert fresh.database_path != loaded_path
    assert loaded.database_path is None and not loaded.database_available
    loaded.close()
    fresh.close()


def test_borrowed_connection_cannot_be_replaced_by_cache(config, tmp_path):
    with sqlite3.connect(":memory:") as connection:
        sim = pax.Simulation(config, connection=connection)
        with pytest.raises(ValueError, match="caller-owned"):
            sim.run(cache_dir=tmp_path / "cache")
        sim.close()
        assert connection.execute("SELECT 1").fetchone() == (1,)
    connection.close()


def test_single_run_rejects_close_during_callback(config):
    sim = pax.Simulation(config)
    attempted = []

    def callback(running):
        with pytest.raises(RuntimeError, match="active"):
            running.close()
        attempted.append(True)

    sim.begin_sample_callback(callback)
    result = sim.run()
    assert attempted and result.database_available
    result.close()


def test_policy_changes_before_run_and_between_runs(config):
    sim = pax.Simulation(config)
    config.db.result_storage = "memory"
    first = sim.run()
    assert first.database_path is None and first.database_available
    config.db = None
    second = sim.run()
    assert second.db is None
    assert first.database_available
    first.close()


@pytest.mark.parametrize("single_process", [True, False])
def test_experiments_allow_distinct_exact_destinations(config, tmp_path, single_process):
    experiments = Experiments(config, output_dir=tmp_path / "outputs")

    @experiments
    def baseline(cfg):
        cfg.db.filename = tmp_path / "baseline.sqlite"
        return cfg

    @experiments
    def alternative(cfg):
        cfg.db.filename = tmp_path / "alternative.sqlite"
        return cfg

    results = experiments.run(single_process=single_process, use_existing=False, write_report=False)
    for name, result in results.items():
        assert result.database_path == tmp_path / f"{name}.sqlite"
        result.close()
        assert result.database_path.exists()


@pytest.mark.parametrize("execution", ["single", "multi"])
def test_exact_destination_cache_hit_never_reopens_original(config, tmp_path, execution):
    config.db.filename = tmp_path / "original.sqlite"
    cls = pax.Simulation if execution == "single" else pax.MultiSimulation
    first = cls(config).run(cache_dir=tmp_path / "cache")
    first.close()
    before = config.db.filename.read_bytes()
    second = cls(config).run(cache_dir=tmp_path / "cache")
    assert not second.database_available
    assert second.database_path is None
    second.close()
    assert config.db.filename.read_bytes() == before


def test_closing_unused_simulation_does_not_allocate(config, tmp_path):
    config.db.filename = tmp_path / "unused.sqlite"
    with pax.Simulation(config) as sim:
        assert not sim.database_available
        assert sim.database_path == config.db.filename
    assert sim.db is None and not config.db.filename.exists()
    result = sim.run()
    assert result.database_available
    result.close()


def test_sequential_reuses_parallel_cached_summary(config: pax.Config, tmp_path: Path) -> None:
    """Changing execution modes must return the summary held by a cached Future."""
    experiments = Experiments(config, output_dir=tmp_path / "outputs")

    @experiments
    def baseline(cfg: pax.Config) -> pax.Config:
        return cfg

    first = experiments.run(single_process=False, use_existing=False, write_report=False)
    second = experiments.run(single_process=True, write_report=False)
    assert second["baseline"] is first["baseline"]
    assert experiments.experiments[0].cached is first["baseline"]
    assert second["baseline"].database_available
    second["baseline"].close()


@pytest.mark.parametrize("single_process", [True, False])
def test_experiments_relocate_absolute_outputs(config: pax.Config, tmp_path: Path, single_process: bool) -> None:
    """All report destinations must be isolated by tag, even with absolute inputs."""
    config.outputs.disk = tmp_path / "shared.pxsim"
    config.outputs.pickle = tmp_path / "shared.pkl"
    config.outputs.excel = tmp_path / "shared.xlsx"
    config.outputs.html.filename = tmp_path / "shared.html"
    experiments = Experiments(config, output_dir=tmp_path / "outputs")

    @experiments
    def baseline(cfg: pax.Config) -> pax.Config:
        return cfg

    @experiments
    def alternative(cfg: pax.Config) -> pax.Config:
        return cfg

    results = experiments.run(single_process=single_process, use_existing=False, write_report=False)
    for tag, summary in results.items():
        for suffix in ("pxsim", "pkl.lz4", "xlsx", "html"):
            assert len(list((tmp_path / "outputs" / tag).glob(f"shared.*.{suffix}"))) == 1
            assert not list(tmp_path.glob(f"shared*.{suffix}"))
        summary.close()
