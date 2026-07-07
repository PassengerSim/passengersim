import pytest

from passengersim import Simulation, demo_network
from passengersim.config import Config
from passengersim.rm import RmSys, register_rm_system
from passengersim.rm.emsr import ExpectedMarginalSeatRevenue
from passengersim.rm.specialty_systems.fcfs import FirstComeFirstServed  # noqa: F401
from passengersim.rm.standard_forecasting import StandardLegForecast, StandardPathForecast
from passengersim.rm.untruncation import LegDetruncation, PathDetruncation
from passengersim.summaries import SimulationTables

DEFAULT_TOLERANCE = dict(rtol=2e-02, atol=1e-06)


@pytest.fixture(scope="module")
def default_config() -> Config:
    input_file = demo_network("3MKT/08-untrunc-em")
    cfg = Config.from_yaml(input_file)
    cfg.simulation_controls.num_trials = 1
    cfg.simulation_controls.num_samples = 500
    return cfg


def test_default_path_truncation_rules(default_config):
    sim0 = Simulation(default_config)
    sim0.setup_scenario()
    assert len(sim0.paths) == 12
    for p in sim0.paths:
        assert p.truncation_rule == 3


def test_carrier_defined_path_truncation_rules(default_config):
    default_config.carriers["AL1"].truncation_rule = 1
    default_config.carriers["AL2"].truncation_rule = 2
    sim1 = Simulation(default_config)
    sim1.setup_scenario()
    assert len(sim1.paths) == 12
    for p in sim1.paths:
        if p.carrier_name == "AL1":
            assert p.truncation_rule == 1
        elif p.carrier_name == "AL2":
            assert p.truncation_rule == 2
        else:
            raise AssertionError(f"Unexpected carrier {p.carrier_name}")


def test_empty_sim_no_database_summary_tables():
    c = Config()
    c.db = None
    c.outputs._write_no_files()
    s = Simulation(c)
    summary = s.run()
    assert isinstance(summary, SimulationTables)
    assert not summary.cnx.is_open


def test_empty_sim_no_database():
    c = Config()
    c.db = None
    c.outputs._write_no_files()
    s = Simulation(c)
    summary = s.run()
    assert isinstance(summary, SimulationTables)
    assert not summary.cnx.is_open


def test_automatic_leg_ids():
    from passengersim.config import Config

    carrier1 = dict(name="X1", control="none", rm_system=FirstComeFirstServed)
    carrier2 = dict(name="X2", control="none", rm_system=FirstComeFirstServed)
    leg1 = dict(
        orig="A",
        dest="B",
        carrier="X1",
        fltno=123,
        dep_time="08:00",
        arr_time="10:00",
        capacity=100,
    )
    # leg2 has different carrier but the same fltno as leg1,
    # so it should get a new leg_id (1)
    leg2 = dict(
        orig="A",
        dest="B",
        carrier="X2",
        fltno=123,
        dep_time="08:00",
        arr_time="10:00",
        capacity=100,
    )

    # These is not really needed for the tests, but it avoids config validation warnings
    places = [
        {"name": "A", "label": "Airport-A", "lat": 33.64, "lon": -84.43},
        {"name": "B", "label": "Airport-B", "lat": 42.3656, "lon": -71.0098},
    ]

    raw = {"legs": [leg1, leg2], "carriers": [carrier1, carrier2], "dcps": [7, 3, 1], "places": places}
    cfg = Config.model_validate(raw)
    sim = Simulation(cfg)
    assert len(sim.legs) == 2
    assert list(sim.legs.keys()) == [123, 1]
    for leg, leg_id in zip(sim.legs, [123, 1]):
        assert leg == leg_id
    assert sim.eng.legs[0].leg_id == 123
    assert sim.eng.legs[1].leg_id == 1


def test_rm_sys_variant():

    from passengersim.rm import make_rm_system_variant
    from passengersim.rm.q_forecasting import QPathForecast
    from passengersim.rm.standard_systems import Qu

    @make_rm_system_variant
    class Qu25_variant(Qu):
        fare_adjustment_scale = 0.212
        fare_adjustment = "ki"

    cfg = Config.from_yaml(demo_network("3MKT/DEMO"))
    cfg.carriers.AL1.rm_system = "Qu25_variant"
    cfg.carriers.AL1.frat5 = "curve_C"
    cfg.carriers.AL1.store_q_history = True

    cfg = Config.model_validate(cfg)
    sim = Simulation(cfg)

    assert isinstance(sim.carriers_dict["AL1"].rm_sys.action_queue[2], QPathForecast)
    assert sim.carriers_dict["AL1"].rm_sys.action_queue[2].fare_adjustment == "ki"
    assert sim.carriers_dict["AL1"].rm_sys.action_queue[2].fare_adjustment_scale == 0.212


def test_mismatched_steps():

    @register_rm_system
    class Bad1(RmSys):
        availability_control = "bp"
        actions = [
            LegDetruncation,
            StandardLegForecast,
            ExpectedMarginalSeatRevenue,
        ]

    cfg = Config.from_yaml(demo_network("3MKT/DEMO"))

    cfg.carriers.AL1.rm_system = "Bad1"
    cfg = Config.model_validate(cfg)
    with pytest.raises(ValueError, match="requires bid_prices for availability control 'bp'"):
        _ = Simulation(cfg)

    @register_rm_system
    class Bad2(RmSys):
        availability_control = "bp"
        actions = [
            LegDetruncation,
            StandardPathForecast,
            ExpectedMarginalSeatRevenue,
        ]

    cfg.carriers.AL1.rm_system = "Bad2"
    cfg = Config.model_validate(cfg)
    with pytest.raises(ValueError, match="Bad2 action ExpectedMarginalSeatRevenue requires 'leg_forecast'"):
        _ = Simulation(cfg)

    @register_rm_system
    class Bad3(RmSys):
        availability_control = "bp"
        actions = [
            PathDetruncation,
            StandardPathForecast,
            ExpectedMarginalSeatRevenue,
        ]

    cfg.carriers.AL1.rm_system = "Bad3"
    cfg = Config.model_validate(cfg)
    with pytest.raises(ValueError, match="action ExpectedMarginalSeatRevenue requires 'leg_forecast'"):
        _ = Simulation(cfg)
