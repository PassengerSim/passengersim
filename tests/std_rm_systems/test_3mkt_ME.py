import hashlib
import json

import altair as alt
from pytest import fixture, mark

import passengersim as pax


@fixture(scope="module", params=[10, 0])
def config(request) -> pax.Config:
    max_cap = request.param
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT"))
    cfg.carriers.AL1.rm_system = "M"
    cfg.carriers.AL2.rm_system = "E"
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 250
    cfg = cfg.model_revalidate()
    cfg.rm_systems.M.processes.DCP.forecast.max_cap = max_cap
    return cfg


@fixture(scope="module")
def summary(config: pax.Config) -> pax.SimulationTables:
    sim = pax.Simulation(config)
    return sim.run(summarizer=pax.SimulationTables)


def _target(basename, maxcap):
    if maxcap not in {0, 10}:
        return basename + f"_max_cap_{maxcap}"
    return basename


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
    max_cap = summary.config.rm_systems.M.processes.DCP.forecast.max_cap
    dataframe_regression.check(df, basename=_target(table_name, max_cap))


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
    ("fig_path_forecasts", dict(by_path_id=5, of=["mu", "sigma", "closed"])),
    ("fig_path_forecasts", dict(by_path_id=9, of=["mu", "sigma", "closed"])),
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
    max_cap = summary.config.rm_systems.M.processes.DCP.forecast.max_cap
    dataframe_regression.check(df, basename=_target(fig_name, max_cap))
