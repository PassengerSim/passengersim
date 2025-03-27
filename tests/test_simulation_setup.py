import pytest

from passengersim import Simulation, demo_network
from passengersim.config import Config
from passengersim.summaries import SimulationTables
from passengersim.summary import SummaryTables

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
    s = Simulation(c)
    summary = s.run(summarizer=None)
    assert isinstance(summary, SummaryTables)
    assert not summary.cnx.is_open


def test_empty_sim_no_database():
    c = Config()
    c.db = None
    s = Simulation(c)
    summary = s.run()
    assert isinstance(summary, SimulationTables)
    assert not summary.cnx.is_open


def test_automatic_leg_ids():
    from passengersim.config import Config

    carrier1 = dict(name="X1", control="none", rm_system="fcfs")
    carrier2 = dict(name="X2", control="none", rm_system="fcfs")
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
    fcfs = dict(availability_control="leg", processes={})
    raw = {
        "legs": [leg1, leg2],
        "carriers": [carrier1, carrier2],
        "rm_systems": {"fcfs": fcfs},
    }
    cfg = Config.model_validate(raw)
    sim = Simulation(cfg)
    assert len(sim.legs) == 2
    assert list(sim.legs.keys()) == [123, 1]
    for leg, leg_id in zip(sim.legs, [123, 1]):
        assert leg == leg_id
    assert sim.sim.legs[0].leg_id == 123
    assert sim.sim.legs[1].leg_id == 1


def test_rm_system_attribute_editing():
    cfg = Config.from_yaml(demo_network("3MKT"))
    cfg.carriers.AL1.rm_system = "L"
    cfg = cfg.model_revalidate()
    assert len(cfg.rm_systems["L"].processes["dcp"]) == 4
    assert cfg.rm_systems.L.processes.dcp[0].step_type == "legvalue"
    cfg.rm_systems.L.processes.dcp = cfg.rm_systems.L.processes.dcp[1:]
    assert len(cfg.rm_systems["L"].processes["dcp"]) == 3
    assert cfg.rm_systems.L.processes.dcp[0].step_type == "untruncation"
