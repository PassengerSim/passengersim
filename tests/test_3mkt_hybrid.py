import pytest

from passengersim import MultiSimulation, Simulation, demo_network
from passengersim.config import Config
from passengersim.summaries import SimulationTables

DEFAULT_TOLERANCE = dict(rtol=1e-04, atol=1e-06)


@pytest.fixture(scope="module")
def config() -> Config:
    input_file = demo_network("3MKT/13-hybrid-fcst")
    cfg = Config.from_yaml(input_file)
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 150
    cfg.simulation_controls.burn_samples = 100
    cfg.outputs.reports.clear()
    cfg.outputs.reports.add("*")
    return cfg


@pytest.fixture(scope="module")
def summary(config: Config) -> SimulationTables:
    sim = Simulation(config)
    return sim.run(summarizer=SimulationTables)


@pytest.fixture(scope="module")
def summary_mp(config: Config) -> SimulationTables:
    sim = MultiSimulation(config)
    return sim.run(summarizer=SimulationTables)


@pytest.fixture(scope="module", params=["ki", "mr"])
def summary_fare_adjustment(
    config: Config, request: pytest.FixtureRequest
) -> (str, SimulationTables):
    fare_adj = request.param
    config.rm_systems.rm_hybrid.processes.dcp.forecast.fare_adjustment = fare_adj
    sim = Simulation(config)
    return fare_adj, sim.run(summarizer=SimulationTables)


TABLES = [
    "demand_to_come",
    "demand_to_come_summary",
    "fare_class_mix",
    "legs",
    "legbuckets",
    "paths",
    "carriers",
    "segmentation_by_timeframe",
    "pathclasses",
    "leg_forecasts",
    "path_forecasts",
]


@pytest.mark.parametrize("table_name", TABLES)
def test_3mkt_hybrid_table_single_process(
    summary, dataframe_regression, table_name: str
):
    assert isinstance(summary, SimulationTables)
    df = getattr(summary, table_name)
    dataframe_regression.check(
        df, basename=table_name, default_tolerance=DEFAULT_TOLERANCE
    )


@pytest.mark.parametrize("table_name", TABLES)
def test_3mkt_hybrid_table_multi_process(
    summary_mp, dataframe_regression, table_name: str
):
    assert isinstance(summary_mp, SimulationTables)
    df = getattr(summary_mp, table_name)
    dataframe_regression.check(
        df, basename=table_name, default_tolerance=DEFAULT_TOLERANCE
    )


@pytest.mark.parametrize("table_name", TABLES)
def test_3mkt_hybrid_fareadj_table_single_process(
    summary_fare_adjustment, dataframe_regression, table_name: str
):
    fare_adj, run_summary = summary_fare_adjustment
    assert isinstance(run_summary, SimulationTables)
    df = getattr(run_summary, table_name)
    dataframe_regression.check(
        df, basename=f"{fare_adj}-{table_name}", default_tolerance=DEFAULT_TOLERANCE
    )
