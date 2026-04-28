import pytest
from pytest import approx

from passengersim import MultiSimulation, Simulation, demo_network
from passengersim.config import Config
from passengersim.summaries import SimulationTables

DEFAULT_TOLERANCE = dict(rtol=1e-04, atol=1e-06)
TABLE_TOLERANCE = {"local_and_flow_yields": dict(rtol=1e-02, atol=1e-06)}


@pytest.fixture(scope="module")
def config(tmp_path_factory) -> Config:
    input_file = demo_network("3MKT/08-untrunc-em")
    cfg = Config.from_yaml(input_file)
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 150
    cfg.simulation_controls.burn_samples = 100
    cfg.simulation_controls.connection_builder.nonstop_leg_path_id_alignment = False
    cfg.outputs.reports.clear()
    cfg.outputs.reports.add("carrier_history")
    cfg.outputs.reports.add("demand_to_come")
    cfg.outputs.reports.add("local_and_flow_yields")
    tmp_path = tmp_path_factory.mktemp("test_summaries")
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg.db.filename = tmp_path
    return cfg


@pytest.fixture(scope="module")
def summary(config: Config, tmp_path_factory) -> SimulationTables:
    tmp_path = tmp_path_factory.mktemp("test_summaries")
    config.db.filename = tmp_path / "db0.sqlite"
    config.outputs.base_dir = tmp_path / "outputs"
    sim = Simulation(config)
    return sim.run(summarizer=SimulationTables)


@pytest.fixture(scope="module")
def summary2(config: Config, tmp_path_factory) -> SimulationTables:
    tmp_path = tmp_path_factory.mktemp("test_summaries")
    config.db.filename = tmp_path / "db1.sqlite"
    sim0 = Simulation(config)
    sim0.run_trial(0)
    config.db.filename = tmp_path / "db2.sqlite"
    config.outputs.base_dir = tmp_path / "outputs"
    sim1 = Simulation(config)
    sim1.run_trial(1)
    return SimulationTables.aggregate(
        [
            SimulationTables.extract(sim0).run_queries(items=config.outputs.reports),
            SimulationTables.extract(sim1).run_queries(items=config.outputs.reports),
        ]
    )


@pytest.fixture(scope="module")
def summary_mp(config: Config, tmp_path_factory) -> SimulationTables:
    tmp_path = tmp_path_factory.mktemp("test_summaries")
    config.db.filename = tmp_path / "dbMP.sqlite"
    config.outputs.base_dir = tmp_path / "outputs"
    sim = MultiSimulation(config)
    return sim.run(summarizer=SimulationTables)


def test_table_basic(summary: SimulationTables):
    assert isinstance(summary, SimulationTables)
    assert isinstance(summary.sim, Simulation)
    assert isinstance(summary.config, Config)

    # for each leg, check that the sum of gt_sold for all paths equals the leg's gt_sold
    for leg in summary.sim.eng.legs:
        local_sold, total_sold = 0, 0
        for pth in summary.sim.eng.paths:
            if pth.get_leg_id(0) == leg.leg_id:
                if pth.num_legs() == 1:
                    local_sold += pth.gt_sold
                total_sold += pth.gt_sold
            if pth.num_legs() == 2:
                if pth.get_leg_id(1) == leg.leg_id:
                    total_sold += pth.gt_sold
        assert leg.gt_sold == total_sold
        assert leg.gt_sold_local == local_sold
        assert leg.avg_local() == (local_sold / total_sold) * 100

    # check that buckets have static fares attached
    assert [b.fcst_revenue for b in summary.sim.legs[101].buckets] == [
        400.0,
        300.0,
        200.0,
        150.0,
        125.0,
        100.0,
    ]

    # check that total revenue aligns over all tables
    carrier_total_revenue = summary.carriers.avg_rev.sum() * summary.n_total_samples
    assert summary.legs.gt_revenue.sum() == approx(carrier_total_revenue)
    assert summary.paths.gt_revenue.sum() == approx(carrier_total_revenue)
    assert summary.pathclasses.gt_revenue.sum() == approx(carrier_total_revenue)
    assert summary.legbuckets.gt_revenue.sum() == approx(carrier_total_revenue)

    # check that total sold aligns over all tables
    carrier_total_sold = summary.carriers.avg_sold.sum() * summary.n_total_samples
    legs_sold = summary.legs.gt_sold.sum()
    assert legs_sold > carrier_total_sold
    # legs sold should be > carrier sold due to connecting itineraries
    assert summary.paths.gt_sold.sum() == approx(carrier_total_sold)
    assert summary.pathclasses.gt_sold.sum() == approx(carrier_total_sold)
    assert summary.legbuckets.gt_sold.sum() == approx(legs_sold)
    assert summary.demands.gt_sold.sum() == approx(carrier_total_sold)

    # check sold and nogo is consistent over demands and segmentation
    dmd_sold = summary.demands.gt_sold.sum()
    n_trials = summary.segmentation_by_timeframe.index.get_level_values("trial").nunique()
    segm_sold = (
        summary.segmentation_by_timeframe.stack("segment", future_stack=True)[["bookings"]]
        .query("carrier != 'NONE'")
        .sum()
        * summary.n_total_samples
        / n_trials
    )
    assert dmd_sold == approx(segm_sold)

    assert "gt_eliminated_no_offers" in summary.demands.columns

    dmd_nogo = summary.demands.eval("gt_eliminated_no_offers + gt_eliminated_chose_nothing + gt_eliminated_wtp").sum()
    segm_nogo = (
        summary.segmentation_by_timeframe.stack("segment", future_stack=True)[["bookings"]]
        .query("carrier == 'NONE'")
        .sum()
        * summary.n_total_samples
        / n_trials
    )
    assert dmd_nogo == pytest.approx(segm_nogo)


TABLES = [
    "demands",
    "demand_to_come",
    "demand_to_come_summary",
    ##    "edgar",
    "fare_class_mix",
    "legs",
    "legbuckets",
    "paths",
    "carriers",
    "carrier_history",
    "segmentation_by_timeframe",
    "pathclasses",
    "bid_price_history",
    "displacement_history",
    "local_and_flow_yields",
    "local_fraction_by_place",
]


@pytest.mark.parametrize("table_name", TABLES)
def test_table_presence_single_process(summary, dataframe_regression, table_name: str):
    assert isinstance(summary, SimulationTables)
    df = getattr(summary, table_name)
    if df.columns.nlevels > 1:
        df.columns = ["__".join(col).strip() for col in df.columns.values]
    dataframe_regression.check(
        df,
        basename=table_name,
        default_tolerance=TABLE_TOLERANCE.get(table_name, DEFAULT_TOLERANCE),
    )


@pytest.mark.parametrize("table_name", TABLES)
def test_table_presence_two_process(summary2, dataframe_regression, table_name: str):
    assert isinstance(summary2, SimulationTables)
    df = getattr(summary2, table_name)
    if df.columns.nlevels > 1:
        df.columns = ["__".join(col).strip() for col in df.columns.values]
    if "trial" in df.index.names:
        df = df.sort_index()
        print("\nSorted by INDEX trial for table:", table_name)
    if "trial" in df.columns:
        df = df.sort_values("trial").reset_index(drop=True)
        print("\nSorted by trial for table:", table_name)
    else:
        print("\nNOT Sorted by trial for table:", table_name)
    print("%%%%% DataFrame info and head %%%%%")
    print(df.info())
    print("%%%%% DataFrame HEAD %%%%%")
    print(df.head())
    print("%%%%% DataFrame TAIL %%%%%")
    print(df.tail())

    dataframe_regression.check(
        df,
        basename=table_name,
        default_tolerance=TABLE_TOLERANCE.get(table_name, DEFAULT_TOLERANCE),
    )


@pytest.mark.parametrize("table_name", TABLES)
def test_table_presence_multi_process(summary_mp, dataframe_regression, table_name: str):
    assert isinstance(summary_mp, SimulationTables)
    df = getattr(summary_mp, table_name)
    if df.columns.nlevels > 1:
        df.columns = ["__".join(col).strip() for col in df.columns.values]
    dataframe_regression.check(
        df,
        basename=table_name,
        default_tolerance=TABLE_TOLERANCE.get(table_name, DEFAULT_TOLERANCE),
    )
