import pytest

import passengersim as pax
from passengersim.rm.emsr import ExpectedMarginalSeatRevenue
from passengersim.rm.heuristic_adjustments.booked_lf import (
    BookedLoadFactorAdjustment,
    collect_blf_detail,
    map_leg_blf_groups,
    process_blf_detail,
)
from passengersim.rm.standard_forecasting import StandardLegForecast
from passengersim.rm.systems import RmSys, RmSysOption, register_rm_system
from passengersim.rm.untruncation import LegDetruncation


@register_rm_system
class EX(RmSys):
    availability_control = "leg"
    """This RM system uses leg-level class allocation availability controls."""

    actions = [
        LegDetruncation,
        StandardLegForecast.configure(
            algorithm=RmSysOption("forecast_algorithm", default="additive_pickup"),
            alpha=RmSysOption("exp_smoothing_alpha", expected_type=float, default=0.15),
        ),
        BookedLoadFactorAdjustment,
        ExpectedMarginalSeatRevenue.configure(
            variant=RmSysOption("emsr_variant", default="b"),
        ),
    ]


@pytest.fixture(scope="session")
def summary():
    cfg = pax.Config.from_yaml(pax.demo_network("3MKT/DEMO"))
    for leg in cfg.legs:
        if leg.carrier == "AL1":
            if leg.orig == "BOS":
                leg.tags["BLF_Curve"] = "BOSTON"
            if leg.orig == "ORD":
                leg.tags["BLF_Curve"] = "CHICAGO"
    cfg.carriers.AL1.rm_system = "EX"
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 100
    cfg.simulation_controls.burn_samples = 50
    cfg.simulation_controls.connection_builder.nonstop_leg_path_id_alignment = False
    cfg.outputs._write_no_files()

    sim = pax.Simulation(cfg)
    sim.daily_callback(collect_blf_detail)
    summary = sim.run()
    return summary


TABLES = [
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
]


def test_table_list(summary):
    assert isinstance(summary, pax.SimulationTables)
    assert all(hasattr(summary, table) for table in TABLES)


@pytest.mark.parametrize("table_name", TABLES)
def test_summary_tables(summary, dataframe_regression, table_name: str):
    assert isinstance(summary, pax.SimulationTables)
    df = getattr(summary, table_name)
    dataframe_regression.check(df, basename=table_name)


def test_blf_detail_processing(summary, dataframe_regression):
    groups = map_leg_blf_groups(summary.config)
    assert groups == {101: "BOSTON", 102: "BOSTON", 111: "CHICAGO", 112: "CHICAGO"}
    df = process_blf_detail(summary, groups)
    dataframe_regression.check(df, basename="blf_detail_processed")
