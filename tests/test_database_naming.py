"""Compatibility for the Database wrapper's public name."""

import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from passengersim import MultiSimulation, Simulation
from passengersim.database import Database
from passengersim.summaries.generic import GenericSimulationTables

pytestmark = pytest.mark.filterwarnings("error:.*cnx.*:DeprecationWarning")


@pytest.fixture
def db():
    with_db = Database(filename=":memory:")
    yield with_db
    with_db.close()


@pytest.mark.parametrize("kind", ["single", "multi", "summary"])
def test_attribute_alias_identity_and_warning(db, kind):
    if kind == "summary":
        obj = GenericSimulationTables(db=db)
    elif kind == "single":
        obj = Simulation.__new__(Simulation)
        obj.db = db
    else:
        obj = MultiSimulation.__new__(MultiSimulation)
        obj._trial_database_store = SimpleNamespace(database=db)
    with pytest.warns(DeprecationWarning, match=r"use .db") as caught:
        assert obj.cnx is obj.db is db
    assert Path(caught[0].filename) == Path(__file__)
    if kind != "multi":
        with pytest.warns(DeprecationWarning) as caught:
            obj.cnx = None
        assert Path(caught[0].filename) == Path(__file__)
        assert obj.db is None
    else:
        with pytest.warns(DeprecationWarning), pytest.raises(AttributeError, match="managed"):
            obj.cnx = db
    assert db.is_open


def test_summary_keyword_compatibility(db):
    with pytest.warns(DeprecationWarning, match="use db=") as caught:
        summary = GenericSimulationTables(cnx=db)
    assert summary.db is db
    assert Path(caught[0].filename) == Path(__file__)
    with pytest.warns(DeprecationWarning):
        assert summary.run_queries(cnx=db, items=[]) is summary
    assert summary.run_queries(db, items=[]) is summary
    with pytest.raises(TypeError, match="both"):
        GenericSimulationTables(db=None, cnx=db)
    with pytest.raises(TypeError, match="both"):
        summary.run_queries(db=None, cnx=db)
    with pytest.raises(TypeError, match="multiple values"):
        summary.run_queries(db, cnx=db)


def test_summary_legacy_state_does_not_warn(db):
    summary = GenericSimulationTables(db=db)
    state = summary.__getstate__()
    assert "db" not in state and "cnx" not in state
    for legacy in ({}, {"cnx": db}, {"db": db, "cnx": None}):
        restored = GenericSimulationTables.__new__(GenericSimulationTables)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            restored.__setstate__({**state, **legacy})
        assert restored.db is (db if legacy else None)
        assert "cnx" not in restored.__dict__


@pytest.mark.parametrize("raw", [False, True])
def test_table_helper_accepts_wrapper_or_connection(db, raw):
    from passengersim.database.database import sql_placeholders
    from passengersim.database.tables import create_table_configs

    target = db.connection if raw else db
    create_table_configs(db=target)
    with pytest.warns(DeprecationWarning, match="use db=") as caught:
        create_table_configs(cnx=target)
    assert Path(caught[0].filename) == Path(__file__)
    assert sql_placeholders(target, 2) == "?, ?"
    with pytest.warns(DeprecationWarning):
        assert sql_placeholders(cnx=target, n=2) == "?, ?"
    with pytest.raises(TypeError, match="both"):
        create_table_configs(db=target, cnx=target)


def test_query_helper_keyword_compatibility(db):
    from passengersim.database.common_queries import total_demand

    db.execute(
        "INSERT INTO demand_detail "
        "(scenario, iteration, trial, sample, days_prior, segment, orig, dest, sample_demand) "
        "VALUES ('test', 0, 0, 1, 0, 'business', 'AAA', 'BBB', 42)"
    )
    assert total_demand(db=db, scenario="test", burn_samples=0) == 42
    with pytest.warns(DeprecationWarning, match="use db="):
        assert total_demand(cnx=db, scenario="test", burn_samples=0) == 42
    with pytest.raises(TypeError, match="both"):
        total_demand(db=db, cnx=db, scenario="test")


@pytest.fixture
def config(tmp_path):
    import passengersim as pax
    from passengersim.config.database import DatabaseConfig

    cfg = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 3
    cfg.simulation_controls.burn_samples = 1
    cfg.simulation_controls.show_progress_bar = False
    cfg.outputs._write_no_files()
    cfg.outputs.reports.clear()
    cfg.db = DatabaseConfig(result_directory=tmp_path)
    return cfg


@pytest.mark.parametrize("execution", ["single", "sequential", "parallel"])
def test_live_alias_shares_managed_database(config, tmp_path, execution):
    sim = Simulation(config) if execution == "single" else MultiSimulation(config)
    assert not list(tmp_path.iterdir())
    assert not sim.database_available
    if execution == "sequential":
        result = sim.sequential_run()
    elif execution == "parallel":
        result = sim.run(max_processes=2)
    else:
        result = sim.run()
    path = result.database_path
    assert path.is_file()
    db = result.db
    assert db is sim.db
    with pytest.warns(DeprecationWarning):
        assert result.cnx is db
    with pytest.warns(DeprecationWarning):
        assert sim.cnx is db
    assert db.execute("SELECT count(DISTINCT trial) FROM leg_detail").fetchone() == (2,)
    result.close()
    assert not db.is_open
    assert not path.exists()
    assert not sim.database_available
    sim.close()


def test_alias_first_access_opens_only_one_database(config, tmp_path):
    with Simulation(config) as sim:
        assert not list(tmp_path.iterdir())
        with pytest.warns(DeprecationWarning):
            db = sim.cnx
        assert db is sim.db
        assert len(list(tmp_path.rglob("*.sqlite"))) == 1


def test_generated_requery_keyword_warning_points_to_caller(db):
    from passengersim.summaries.generic import DatabaseTableItem

    def query(db, **kwargs):
        return db.dataframe("SELECT 42 AS answer")

    class Summary(GenericSimulationTables):
        # Keep this test's descriptor registrations separate from other summaries.
        _std_agg = {}
        _std_query = {}
        answer = DatabaseTableItem(query_func=query, aggregation_func=None)

    summary = Summary(db=db)
    assert summary._requery_answer(db=db).iloc[0, 0] == 42
    with pytest.warns(DeprecationWarning) as caught:
        assert summary._requery_answer(cnx=db).iloc[0, 0] == 42
    assert Path(caught[0].filename) == Path(__file__)
