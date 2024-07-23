
from typing import Literal

import altair
import pytest

from passengersim import Simulation, demo_network
from passengersim.config import Config
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
    default_config.airlines['AL1'].truncation_rule = 1
    default_config.airlines['AL2'].truncation_rule = 2
    sim1 = Simulation(default_config)
    sim1.setup_scenario()
    assert len(sim1.paths) == 12
    for p in sim1.paths:
        if p.carrier == 'AL1':
            assert p.truncation_rule == 1
        elif p.carrier == 'AL2':
            assert p.truncation_rule == 2
        else:
            raise AssertionError(f"Unexpected carrier {p.carrier}")
