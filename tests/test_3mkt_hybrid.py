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


@pytest.fixture(scope="module", params=[None, "em"])
def summary(
    config: Config, request: pytest.FixtureRequest
) -> (str | None, SimulationTables):
    detruncation = request.param
    if detruncation:
        pytest.skip("hybrid with detruncation testing is suspended")
    config.rm_systems.rm_hybrid.processes.dcp.forecast.detruncation_algorithm = (
        detruncation
    )
    sim = Simulation(config)
    return detruncation, sim.run(summarizer=SimulationTables)


@pytest.fixture(scope="module", params=[None, "em"])
def summary_mp(
    config: Config, request: pytest.FixtureRequest
) -> (str | None, SimulationTables):
    detruncation = request.param
    if detruncation:
        pytest.skip("hybrid with detruncation testing is suspended")
    config.rm_systems.rm_hybrid.processes.dcp.forecast.detruncation_algorithm = (
        detruncation
    )
    sim = MultiSimulation(config)
    return detruncation, sim.run(summarizer=SimulationTables)


@pytest.fixture(
    scope="module",
    params=[
        ("ki", None, 1.0),
        ("mr", None, 1.0),
        ("ki", "em", 1.0),
        ("mr", "em", 1.0),
        ("ki", "em", 0.5),
        ("mr", "em", 0.5),
    ],
)
def summary_fare_adjustment(
    config: Config, request: pytest.FixtureRequest
) -> (str, SimulationTables):
    fare_adj = request.param[0]
    detruncation = request.param[1]
    if detruncation:
        pytest.skip("Fare adjustment with detruncation testing is suspended")
    fare_adj_scale = request.param[2]
    config.rm_systems.rm_hybrid.processes.dcp.forecast.fare_adjustment = fare_adj
    config.rm_systems.rm_hybrid.processes.dcp.forecast.fare_adjustment_scale = (
        fare_adj_scale
    )
    config.rm_systems.rm_hybrid.processes.dcp.forecast.detruncation_algorithm = (
        detruncation
    )
    sim = Simulation(config)
    return fare_adj, detruncation, fare_adj_scale, sim.run(summarizer=SimulationTables)


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

TABLE_NOT_SENSITIVE_TO_RM = [
    "demand_to_come",
    "demand_to_come_summary",
]


@pytest.mark.parametrize("table_name", TABLES)
def test_3mkt_hybrid_table_single_process(
    summary, dataframe_regression, table_name: str
):
    detruncation, summary = summary
    assert isinstance(summary, SimulationTables)
    df = getattr(summary, table_name)
    if table_name in TABLE_NOT_SENSITIVE_TO_RM:
        basename = table_name
    else:
        basename = f"{detruncation}-{table_name}"
    dataframe_regression.check(
        df, basename=basename, default_tolerance=DEFAULT_TOLERANCE
    )


@pytest.mark.parametrize("table_name", TABLES)
def test_3mkt_hybrid_table_multi_process(
    summary_mp, dataframe_regression, table_name: str
):
    detruncation, summary_mp = summary_mp
    assert isinstance(summary_mp, SimulationTables)
    df = getattr(summary_mp, table_name)
    if table_name in TABLE_NOT_SENSITIVE_TO_RM:
        basename = table_name
    else:
        basename = f"{detruncation}-{table_name}"
    dataframe_regression.check(
        df, basename=basename, default_tolerance=DEFAULT_TOLERANCE
    )


@pytest.mark.parametrize("table_name", TABLES)
def test_3mkt_hybrid_fareadj_table_single_process(
    summary_fare_adjustment, dataframe_regression, table_name: str
):
    fare_adj, detruncation, fare_adj_scale, run_summary = summary_fare_adjustment
    assert isinstance(run_summary, SimulationTables)
    df = getattr(run_summary, table_name)
    if table_name in TABLE_NOT_SENSITIVE_TO_RM:
        basename = table_name
    else:
        basename = f"{fare_adj}-{fare_adj_scale}-{detruncation}-{table_name}"
    dataframe_regression.check(
        df, basename=basename, default_tolerance=DEFAULT_TOLERANCE
    )
