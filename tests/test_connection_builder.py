import pytest
from pytest import raises

from passengersim import Simulation, demo_network
from passengersim.config import Config

DEFAULT_TOLERANCE = dict(rtol=2e-02, atol=1e-06)


@pytest.fixture
def cfg() -> Config:
    input_file = demo_network("3MKT/08-untrunc-em")
    cfg = Config.from_yaml(input_file)
    assert cfg.paths is not None
    assert len(cfg.paths) == 0
    assert cfg.places is not None
    assert len(cfg.places) == 3
    return cfg


def test_bad_mct(cfg):
    with raises(ValueError):
        cfg.places["ORD"].mct = "nope"


def test_simple_mct(cfg):
    cfg.places["ORD"].mct = 60
    assert cfg.places["ORD"].mct.domestic_domestic == 60
    assert cfg.places["ORD"].mct.domestic_international == 60
    assert cfg.places["ORD"].mct.international_domestic == 60
    assert cfg.places["ORD"].mct.international_international == 60
    sim = Simulation(cfg)
    sim.sim.build_connections()
    assert len(sim.paths) == 12


def test_quad_list_mct(cfg):
    cfg.places["ORD"].mct = [60, 90, 120, 140]
    assert cfg.places["ORD"].mct.domestic_domestic == 60
    assert cfg.places["ORD"].mct.domestic_international == 90
    assert cfg.places["ORD"].mct.international_domestic == 120
    assert cfg.places["ORD"].mct.international_international == 140
    sim = Simulation(cfg)
    sim.sim.build_connections()
    assert len(sim.paths) == 12


def test_shorthand_mct(cfg):
    cfg.places["ORD"].mct = dict(dd=60, di=90, id=120, ii=140)
    assert cfg.places["ORD"].mct.domestic_domestic == 60
    assert cfg.places["ORD"].mct.domestic_international == 90
    assert cfg.places["ORD"].mct.international_domestic == 120
    assert cfg.places["ORD"].mct.international_international == 140
    sim = Simulation(cfg)
    sim.sim.build_connections()
    assert len(sim.paths) == 12


def test_extreme_mct(cfg):
    cfg.places["ORD"].mct = 1440
    sim = Simulation(cfg)
    sim.sim.build_connections()
    # no connections are possible with a 24 hour MCT, but
    # every leg is a non-stop path, so there are still 8 paths
    assert len(sim.paths) == 8
