import hashlib
import json

import altair as alt
from pytest import fixture, mark

import passengersim as pax


@fixture(
    scope="module",
    params=[
        dict(max_cap=10, fa="mr", scale=0.25),
        dict(max_cap=10, fa="ki", scale=0.25),
        dict(max_cap=2, fa="mr", scale=0.25),
        dict(max_cap=10, fa="mr", scale=0.50),
        dict(max_cap=10, fa="mr", scale=1.0),
    ],
)
def config(request) -> pax.Config:
    print("\n\nRunning test with parameters:", request.param, "\n")
    max_cap = request.param.get("max_cap", 10)
    fa_algo = request.param.get("fa", None)
    fa_scale = request.param.get("scale", 0.0)
    # if os.getenv("GITHUB_ACTIONS") == "true":
    #     skip("Skipping on GitHub Actions")
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))
    cfg.simulation_controls.connection_builder.nonstop_leg_path_id_alignment = True
    cfg.carriers.AL1.rm_system = "Qu"
    cfg.carriers.AL1.rm_system_options = {
        "name": "Qu",
        "max_cap": max_cap,
        "fare_adjustment": fa_algo,
        "fare_adjustment_scale": fa_scale,
    }
    cfg.carriers.AL1.frat5 = "curve_C"
    cfg.carriers.AL1.store_q_history = True
    cfg._load_std_frat5("curve_C")
    cfg.frat5_curves["curve_C"].max_cap = max_cap
    cfg.carriers.AL2.rm_system_options = {"name": "E"}
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 250
    cfg.outputs._write_no_files()
    return cfg


def _target(basename, maxcap, fa_algo=None, fa_scale=None):
    basename = f"{basename}_maxcap{maxcap}_{fa_algo}_{int(fa_scale * 100)}"
    return basename


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
    "path_forecasts",
]


def test_table_list(summary):
    assert isinstance(summary, pax.SimulationTables)
    assert all(hasattr(summary, table) for table in TABLES)


@mark.parametrize("table_name", TABLES)
def test_summary_tables(summary, dataframe_regression, table_name: str):
    assert isinstance(summary, pax.SimulationTables)
    df = getattr(summary, table_name)
    max_cap = summary.config.carriers.AL1.rm_system_options.get("max_cap", None)
    fa_algo = summary.config.carriers.AL1.rm_system_options.get("fare_adjustment", None)
    fa_scale = summary.config.carriers.AL1.rm_system_options.get("fare_adjustment_scale", None)
    dataframe_regression.check(df, basename=_target(table_name, max_cap, fa_algo, fa_scale))


FIGURES = [
    ("fig_carrier_revenues", {}),
    ("fig_carrier_load_factors", {}),
    ("fig_carrier_yields", {}),
    ("fig_carrier_local_share", {}),
    ("fig_carrier_local_share", dict(load_measure="leg_pax")),
    ("fig_leg_load_v_local", {}),
    ("fig_leg_forecasts", dict(by_leg_id=201, of=["mu", "sigma", "closed"])),
    ("fig_leg_forecasts", dict(by_leg_id=211, of=["mu", "sigma", "closed"])),
    ("fig_path_forecasts", dict(by_path_id=1, of=["mu", "sigma", "closed"])),
    ("fig_path_forecasts", dict(by_path_id=101, of=["mu", "sigma", "closed"])),
    ("fig_path_forecasts", dict(by_path_id=111, of=["mu", "sigma", "closed"])),
    ("fig_od_fare_class_mix", dict(orig="ORD", dest="LAX")),
    ("fig_segmentation_by_timeframe", dict(metric="bookings")),
    ("fig_segmentation_by_timeframe", dict(metric="revenue")),
    (
        "fig_segmentation_by_timeframe",
        dict(metric="bookings", by_carrier="AL1", by_class=True),
    ),
    ("fig_bid_price_history", {}),
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
    max_cap = summary.config.carriers.AL1.rm_system_options.get("max_cap", None)
    fa_algo = summary.config.carriers.AL1.rm_system_options.get("fare_adjustment", None)
    fa_scale = summary.config.carriers.AL1.rm_system_options.get("fare_adjustment_scale", None)
    dataframe_regression.check(df, basename=_target(fig_name, max_cap, fa_algo, fa_scale))
