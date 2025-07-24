import pytest
from passengersim_core import ForecastVectors

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
def summary(config: Config, request: pytest.FixtureRequest) -> (str | None, SimulationTables):
    detruncation = request.param
    if detruncation:
        pytest.skip("hybrid with detruncation testing is suspended")
    # config.rm_systems.rm_hybrid.processes.dcp.forecast.detruncation_algorithm = (
    #     detruncation
    # )
    sim = Simulation(config)
    return detruncation, sim.run(summarizer=SimulationTables)


@pytest.fixture(scope="module", params=[None, "em"])
def summary_mp(config: Config, request: pytest.FixtureRequest) -> (str | None, SimulationTables):
    detruncation = request.param
    if detruncation:
        pytest.skip("hybrid with detruncation testing is suspended")
    # config.rm_systems.rm_hybrid.processes.dcp.forecast.detruncation_algorithm = (
    #     detruncation
    # )
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
def summary_fare_adjustment(config: Config, request: pytest.FixtureRequest) -> (str, SimulationTables):
    fare_adj = request.param[0]
    detruncation = request.param[1]
    if detruncation:
        pytest.skip("Fare adjustment with detruncation testing is suspended")
    fare_adj_scale = request.param[2]
    config.rm_systems.rm_hybrid.processes.dcp.forecast.fare_adjustment = fare_adj
    config.rm_systems.rm_hybrid.processes.dcp.forecast.fare_adjustment_scale = fare_adj_scale
    # config.rm_systems.rm_hybrid.processes.dcp.forecast.detruncation_algorithm = (
    #     detruncation
    # )
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
def test_3mkt_hybrid_table_single_process(summary, dataframe_regression, table_name: str):
    detruncation, summary = summary
    assert isinstance(summary, SimulationTables)
    df = getattr(summary, table_name)
    if table_name in TABLE_NOT_SENSITIVE_TO_RM:
        basename = table_name
    else:
        basename = f"{detruncation}-{table_name}"
    dataframe_regression.check(df, basename=basename, default_tolerance=DEFAULT_TOLERANCE)


@pytest.mark.parametrize("table_name", TABLES)
def test_3mkt_hybrid_table_multi_process(summary_mp, dataframe_regression, table_name: str):
    detruncation, summary_mp = summary_mp
    assert isinstance(summary_mp, SimulationTables)
    df = getattr(summary_mp, table_name)
    if table_name in TABLE_NOT_SENSITIVE_TO_RM:
        basename = table_name
    else:
        basename = f"{detruncation}-{table_name}"
    dataframe_regression.check(df, basename=basename, default_tolerance=DEFAULT_TOLERANCE)


@pytest.mark.parametrize("table_name", TABLES)
def test_3mkt_hybrid_fareadj_table_single_process(summary_fare_adjustment, dataframe_regression, table_name: str):
    fare_adj, detruncation, fare_adj_scale, run_summary = summary_fare_adjustment
    assert isinstance(run_summary, SimulationTables)
    df = getattr(run_summary, table_name)
    if table_name in TABLE_NOT_SENSITIVE_TO_RM:
        basename = table_name
    else:
        basename = f"{fare_adj}-{fare_adj_scale}-{detruncation}-{table_name}"
    dataframe_regression.check(df, basename=basename, default_tolerance=DEFAULT_TOLERANCE)


@pytest.mark.parametrize(
    "fareadj,adjscale",
    [
        ("ki", 1.0),
        ("mr", 1.0),
        ("ki", 0.5),
        ("mr", 0.5),
        ("ki", 0.000001),
        ("mr", 0.000001),
        (None, 1.0),
    ],
)
def test_fare_adj_walk(data_regression, fareadj, adjscale):
    input_file = demo_network("3MKT/13-hybrid-fcst")
    cfg = Config.from_yaml(input_file)
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 40
    cfg.simulation_controls.burn_samples = 30
    cfg.outputs.reports.clear()
    cfg.rm_systems.rm_hybrid.processes.dcp.forecast.fare_adjustment = fareadj
    cfg.rm_systems.rm_hybrid.processes.dcp.forecast.fare_adjustment_scale = adjscale

    self = Simulation(cfg)
    self.setup_scenario()
    trial = 0
    # self.sim.update_db_write_flags()
    self.begin_trial(trial)

    for s in range(cfg.simulation_controls.num_samples):
        with self.run_single_sample():
            state = {}
            if s > 14 and s % 5:
                # after the 15th sample, check every 5th sample
                continue
            for pth in self.sim.paths:
                # select three paths for checking
                if pth.path_id not in [1, 6, 11]:
                    continue
                try:
                    pth_q = pth.q_forecast.get_vectors()
                except AttributeError:
                    pth_q = ForecastVectors([(), (), (), (), (), ()])
                except ValueError:
                    pth_q = ForecastVectors(["NA", "NA", "NA", "NA", "NA", "NA"])
                state[f"Path-{pth.path_id}"] = {
                    "path_level_q_forecast": {
                        "mean_in_timeframe": list(pth_q.mean_in_timeframe),
                        "mean_to_departure": list(pth_q.mean_to_departure),
                        "stdev_in_timeframe": list(pth_q.stdev_in_timeframe),
                        "stdev_to_departure": list(pth_q.stdev_to_departure),
                    },
                }
                for pc in pth.pathclasses:
                    try:
                        q = pc.q_forecast.get_vectors()
                    except ValueError:
                        q = ForecastVectors([(), (), (), (), (), ()])
                    try:
                        y = pc.y_forecast.get_vectors()
                    except ValueError:
                        y = ForecastVectors([(), (), (), (), (), ()])
                    try:
                        t = pc.forecast.get_vectors()
                    except ValueError:
                        t = ForecastVectors([(), (), (), (), (), ()])
                    state[f"Path-{pth.path_id}"][f"Class-{pc.booking_class}"] = {
                        "sold": pc.sold,
                        "sold_priceable": pc.sold_priceable,
                        "adj_fares": list(pc.raw_adjusted_fare_price),
                        "q_forecast": {
                            "mean_in_timeframe": list(q.mean_in_timeframe),
                            "mean_to_departure": list(q.mean_to_departure),
                            "stdev_in_timeframe": list(q.stdev_in_timeframe),
                            "stdev_to_departure": list(q.stdev_to_departure),
                        },
                        "y_forecast": {
                            "mean_in_timeframe": list(y.mean_in_timeframe),
                            "mean_to_departure": list(y.mean_to_departure),
                            "stdev_in_timeframe": list(y.stdev_in_timeframe),
                            "stdev_to_departure": list(y.stdev_to_departure),
                        },
                        "forecast": {
                            "mean_in_timeframe": list(t.mean_in_timeframe),
                            "mean_to_departure": list(t.mean_to_departure),
                            "stdev_in_timeframe": list(t.stdev_in_timeframe),
                            "stdev_to_departure": list(t.stdev_to_departure),
                        },
                    }

            if adjscale < 0.0001 or fareadj is None:
                # change to no fare adjustment for this check, to confirm that
                # scaling fare adjustment all the way to zero is equivalent to
                # no fare adjustment
                fareadj = None
                adjscale = 1.0
                # also remove calculated adjusted fares from the state, there will
                # be insignificant differences we want to ignore
                for pth in self.sim.paths:
                    for pc in pth.pathclasses:
                        try:
                            state[f"Path-{pth.path_id}"][f"Class-{pc.booking_class}"].pop("adj_fares")
                        except KeyError:
                            pass
            for k, v in state.items():
                for kk, vv in v.items():
                    data_regression.check(
                        vv,
                        basename=f"fareadj-walk/{fareadj}-{int(adjscale*100):03d}/" f"Sample{s}/{k}/{kk}",
                        round_digits=6,
                    )

        # # check selected internal results at every sample
        # if s < 10: continue
        # print('\n<<<<< SAMPLE', self.sim.sample, '>>>>>')
        # print("pth-q:", state['Path-1']['path_level_q_forecast']['mean_to_departure'])
        # print("Y0:", state['Path-1']['Class-Y0']['q_forecast']['mean_to_departure'])
        # print("Y1:", state['Path-1']['Class-Y1']['q_forecast']['mean_to_departure'])
        # print("Y4:", state['Path-1']['Class-Y4']['q_forecast']['mean_to_departure'])
        # print("Y5:", state['Path-1']['Class-Y5']['q_forecast']['mean_to_departure'])
        # print("Y0:", state['Path-1']['Class-Y0']['adj_fares'])
        # print("Y1:", state['Path-1']['Class-Y1']['adj_fares'])
        # print("Y4:", state['Path-1']['Class-Y4']['adj_fares'])
        # print("Y5:", state['Path-1']['Class-Y5']['adj_fares'])
