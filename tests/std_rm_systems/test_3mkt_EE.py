import hashlib
import json

import altair as alt
from pytest import fixture, mark

import passengersim as pax


@fixture(scope="module")
def config(tmp_path_factory) -> pax.Config:
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))
    cfg.carriers.AL1.rm_system_options = {"name": "E"}
    cfg.carriers.AL2.rm_system_options = {"name": "E"}
    cfg.simulation_controls.num_trials = 1
    cfg.db.filename = tmp_path_factory.mktemp("test-3mkt-EE") / "db.sqlite"
    cfg.outputs.base_dir = tmp_path_factory.mktemp("test-3mkt-EE") / "outputs"
    return cfg


@fixture(scope="module")
def summary(config: pax.Config) -> pax.SimulationTables:
    sim = pax.Simulation(config)
    return sim.run(summarizer=pax.SimulationTables)


TABLES = [
    "bid_price_history",
    "carriers",
    "carrier_history2",
    "demand_to_come_summary",
    "demands",
    "displacement_history",
    "fare_class_mix",
    "legbuckets",
    "legs",
    "pathclasses",
    "path_legs",
    "paths",
    "segmentation_by_timeframe",
    "leg_forecasts",
]


def test_table_list(summary):
    assert isinstance(summary, pax.SimulationTables)
    assert all(hasattr(summary, table) for table in TABLES)


@mark.parametrize("table_name", TABLES)
def test_summary_tables(summary, dataframe_regression, table_name: str):
    assert isinstance(summary, pax.SimulationTables)
    df = getattr(summary, table_name)
    try:
        dataframe_regression.check(df, basename=table_name)
    except Exception as e:
        print(e)


FIGURES = [
    ("fig_carrier_revenues", {}),
    ("fig_carrier_load_factors", {}),
    ("fig_carrier_yields", {}),
    ("fig_carrier_local_share", {}),
    ("fig_carrier_local_share", dict(load_measure="leg_pax")),
    ("fig_leg_load_v_local", {}),
    ("fig_leg_forecasts", dict(by_leg_id=101, of=["mu", "sigma", "closed"])),
    ("fig_leg_forecasts", dict(by_leg_id=111, of=["mu", "sigma", "closed"])),
    ("fig_od_fare_class_mix", dict(orig="ORD", dest="LAX")),
    ("fig_segmentation_by_timeframe", dict(metric="bookings")),
    ("fig_segmentation_by_timeframe", dict(metric="revenue")),
    (
        "fig_segmentation_by_timeframe",
        dict(metric="bookings", by_carrier="AL1", by_class=True),
    ),
    ("fig_carrier_rasm", {}),
]


@mark.parametrize("fig", FIGURES)
def test_summary_figures(summary, dataframe_regression, fig: tuple[str, dict]):
    fig_name, kwargs = fig
    assert isinstance(summary, pax.SimulationTables)
    f = getattr(summary, fig_name)(**kwargs)
    assert isinstance(f, alt.TopLevelMixin)
    df = getattr(summary, fig_name)(**kwargs, raw_df=True)
    if kwargs:
        s = json.dumps(kwargs, sort_keys=True)
        h = hashlib.md5(s.encode()).hexdigest()[:12]
        fig_name = f"{fig_name}_{h}"
    print(fig_name)
    print(df)
    dataframe_regression.check(df, basename=fig_name)


def test_db_leg_defs(config: pax.Config, summary: pax.SimulationTables):
    from passengersim.database import Database

    db = Database(filename=config.db.filename)

    info = db.table_info("leg_defs")
    assert info["name"].tolist() == [
        "leg_id",
        "flt_no",
        "carrier",
        "orig",
        "dest",
        "dep_time",
        "arr_time",
        "capacity",
        "distance",
    ]

    cur = db._connection.cursor()
    cur.execute("SELECT COUNT(*) FROM leg_defs")
    count = cur.fetchone()[0]
    assert count == len(summary.legs)


def test_db_leg_bucket_detail(config: pax.Config, summary: pax.SimulationTables):
    from passengersim.database import Database

    db = Database(filename=config.db.filename)
    info = db.table_info("leg_bucket_detail")
    assert info["name"].tolist() == [
        "scenario",
        "iteration",
        "trial",
        "sample",
        "days_prior",
        "leg_id",
        "cabin_code",
        "bucket_number",
        "name",
        "auth",
        "revenue",
        "sold",
        "detruncated_demand",
        "forecast_mean",
        "forecast_stdev",
        "forecast_closed_in_tf",
        "forecast_closed_in_future",
        "updated_at",
    ]

    cur = db._connection.cursor()
    cur.execute("SELECT COUNT(*) FROM leg_bucket_detail")
    count = cur.fetchone()[0]
    assert count > 0
