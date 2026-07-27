import pytest
from pytest import raises

from passengersim import Simulation, demo_network
from passengersim.config import Config
from passengersim.connection_builder.prebuild import prebuild_connections

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
    sim.eng.build_connections()
    assert len(sim.paths) == 12


def test_quad_list_mct(cfg):
    cfg.places["ORD"].mct = [60, 90, 120, 140]
    assert cfg.places["ORD"].mct.domestic_domestic == 60
    assert cfg.places["ORD"].mct.domestic_international == 90
    assert cfg.places["ORD"].mct.international_domestic == 120
    assert cfg.places["ORD"].mct.international_international == 140
    sim = Simulation(cfg)
    sim.eng.build_connections()
    assert len(sim.paths) == 12


def test_shorthand_mct(cfg):
    cfg.places["ORD"].mct = dict(dd=60, di=90, id=120, ii=140)
    assert cfg.places["ORD"].mct.domestic_domestic == 60
    assert cfg.places["ORD"].mct.domestic_international == 90
    assert cfg.places["ORD"].mct.international_domestic == 120
    assert cfg.places["ORD"].mct.international_international == 140
    sim = Simulation(cfg)
    sim.eng.build_connections()
    assert len(sim.paths) == 12


def test_extreme_mct(cfg):
    cfg.places["ORD"].mct = 1440
    sim = Simulation(cfg)
    sim.eng.build_connections()
    # no connections are possible with a 24 hour MCT, but
    # every leg is a non-stop path, so there are still 8 paths
    assert len(sim.paths) == 8


def test_repeated_pathbuilding(cfg):
    # No paths exist yet
    assert len(cfg.paths) == 0
    # Build the paths once
    cfg = prebuild_connections(cfg, inplace=False)
    assert len(cfg.paths) == 12
    # Call for building them again, should not create additional paths
    cfg = prebuild_connections(cfg, inplace=False)
    assert len(cfg.paths) == 12


def test_global_min_connect_time_simple(cfg):
    """A single-int global minimum connect time inflates to all four types."""
    cfg.simulation_controls.connection_builder.min_connect_time = 60
    mct = cfg.simulation_controls.connection_builder.min_connect_time
    assert mct.domestic_domestic == 60
    assert mct.domestic_international == 60
    assert mct.international_domestic == 60
    assert mct.international_international == 60


def test_global_min_connect_time_shorthand(cfg):
    """Global minimum connect time accepts DD/DI/ID/II shorthand."""
    cfg.simulation_controls.connection_builder.min_connect_time = dict(dd=25, di=35, id=55, ii=65)
    mct = cfg.simulation_controls.connection_builder.min_connect_time
    assert mct.domestic_domestic == 25
    assert mct.domestic_international == 35
    assert mct.international_domestic == 55
    assert mct.international_international == 65


def test_global_max_connect_time_list(cfg):
    """Global maximum connect time accepts a list in DD, DI, ID, II order."""
    cfg.simulation_controls.connection_builder.max_connect_time = [240, 480, 480, 480]
    mct = cfg.simulation_controls.connection_builder.max_connect_time
    assert mct.domestic_domestic == 240
    assert mct.domestic_international == 480
    assert mct.international_domestic == 480
    assert mct.international_international == 480


def test_global_min_connect_time_fallback(cfg):
    """An extreme global minimum connect time applies to airports with no local value.

    None of the 3MKT places set their own minimum connect time, so the global value
    is used as the fallback.  An extreme value prevents any connections, leaving only
    the nonstop paths.
    """
    cfg.simulation_controls.connection_builder.min_connect_time = 1440
    sim = Simulation(cfg)
    sim.eng.build_connections()
    assert len(sim.paths) == 8


def test_local_min_connect_time_overrides_global(cfg):
    """A local airport minimum connect time overrides the global default."""
    # Extreme global value would otherwise block all connections (see fallback test).
    cfg.simulation_controls.connection_builder.min_connect_time = 1440
    # But the hub (ORD) sets a permissive local value, restoring connections.
    cfg.places["ORD"].mct = 20
    sim = Simulation(cfg)
    sim.eng.build_connections()
    assert len(sim.paths) == 12


def test_global_connect_times_applied_to_engine(cfg):
    """The connection builder globals are pushed through to the SimulationEngine."""
    cfg.simulation_controls.connection_builder.min_connect_time = dict(dd=25, di=35, id=55, ii=65)
    cfg.simulation_controls.connection_builder.max_connect_time = dict(dd=300, di=400, id=500, ii=600)
    sim = Simulation(cfg)
    assert sim.eng.min_connect_time_dd == 25
    assert sim.eng.min_connect_time_di == 35
    assert sim.eng.min_connect_time_id == 55
    assert sim.eng.min_connect_time_ii == 65
    assert sim.eng.max_connect_time_dd == 300
    assert sim.eng.max_connect_time_di == 400
    assert sim.eng.max_connect_time_id == 500
    assert sim.eng.max_connect_time_ii == 600


def test_legacy_max_connect_time_migration():
    """The deprecated simulation_controls.max_connect_time migrates to the new field."""
    from passengersim.config.simulation_controls import SimulationSettings

    with pytest.warns(DeprecationWarning, match="max_connect_time"):
        ss = SimulationSettings(max_connect_time=300)
    # The legacy value is forwarded to connection_builder.max_connect_time.
    assert ss.connection_builder.max_connect_time.domestic_domestic == 300
    assert ss.connection_builder.max_connect_time.international_international == 300


def test_explicit_connection_builder_max_connect_time_wins_over_legacy():
    """An explicit connection_builder.max_connect_time is not overwritten by the legacy field."""
    from passengersim.config.simulation_controls import SimulationSettings

    with pytest.warns(DeprecationWarning, match="max_connect_time"):
        ss = SimulationSettings(
            max_connect_time=300,
            connection_builder=dict(max_connect_time=180),
        )
    assert ss.connection_builder.max_connect_time.domestic_domestic == 180
