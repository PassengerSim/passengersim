import gzip
import io
from pathlib import Path

import pytest
import yaml
from passengersim_core.airline import ForecastStep, UntruncationStep
from pydantic import ValidationError

from passengersim import Config


def test_rm_systems():
    demo1 = """
    rm_systems:
      - name: SystemA
        processes:
          DCP:
          - step_type: untruncation
            name: foo
            algorithm: em
          - step_type: forecast
            name: baz
            algorithm: exp_smoothing
            alpha: 0.42
      - name: SystemB
        processes:
          DCP:
          - step_type: forecast
            name: baz
            algorithm: additive_pickup
    """
    content = yaml.safe_load(io.StringIO(demo1))
    loaded = Config.model_validate(content)

    system0 = loaded.rm_systems["SystemA"]
    assert system0.name == "SystemA"
    assert len(system0.processes["dcp"]) == 2
    assert isinstance(system0.processes["dcp"][0], UntruncationStep)
    assert isinstance(system0.processes["dcp"][1], ForecastStep)

    system1 = loaded.rm_systems["SystemB"]
    assert system1.name == "SystemB"
    assert isinstance(system1.processes["dcp"][0], ForecastStep)
    assert system1.processes["dcp"][0].step_type == "forecast"
    assert system1.processes["dcp"][0].algorithm == "additive_pickup"
    assert system1.processes["dcp"][0].name == "baz"

    # there are several errors in demo2, the parser finds and
    # reports them all with legible error message
    demo2 = """
    rm_systems:
      processes:
        DCP:
        - step_type: untruncation_misspelled
          name: foo
          algorithm: bar
        - step_type: forecast
          algorithm_misspelled: spam
          alpha: 0.42
    """
    content = yaml.safe_load(io.StringIO(demo2))
    with pytest.raises(ValidationError):
        loaded = Config.model_validate(content)


def test_u10_loader():
    u10_config = Path(__file__).parents[2].joinpath("air-sim/networks/u10-config.yaml")
    if not u10_config.exists():
        pytest.skip("u10-config.yaml not available")
    u10_network = (
        Path(__file__).parents[2].joinpath("air-sim/networks/u10-network.yaml.gz")
    )
    if not u10_network.exists():
        pytest.skip("u10-network.yaml.gz not available")
    with open(u10_config) as f:
        content = yaml.safe_load(f)
    with gzip.open(u10_network) as f:
        content.update(yaml.safe_load(f))
    u10 = Config.model_validate(content)
    assert len(u10.airlines) == 4
    assert u10.airlines["AL2"].name == "AL2"


# def test_u10_transcoder():
#     sd = SimDriver(
#         input_file=Path(__file__).parents[2].joinpath("networks/u10-airsim.txt"),
#     )
#     sd.loader.dump_network(sd, "/tmp/u10-temp.yml")


# def test_3mkt_transcoder():
#     sd = SimDriver(
#         input_file=Path(__file__).parents[2].joinpath("networks/3mkt-temp.txt"),
#     )
#     sd.loader.dump_network(sd, "/tmp/3mkt.yaml")


def test_carriers_have_classes():
    demo = """
    airlines:
      AL1:
        rm_system: SystemA
        frat5: curve_G
      AL2:
        rm_system: SystemA
        frat5: curve_1_5
        classes:
          - F
          - C
          - Y
    classes: # global classes assigned to AL1 because it has none of its own
      - Y0
      - Y1
      - Y2
      - Y3
    rm_systems:
      - name: SystemA
        processes: {}
    """
    content = yaml.safe_load(io.StringIO(demo))
    loaded = Config.model_validate(content)
    assert loaded.airlines["AL1"].classes == ["Y0", "Y1", "Y2", "Y3"]
    assert loaded.airlines["AL2"].classes == ["F", "C", "Y"]

def test_format_tags():
    demo = """
    scenario: party
    tags:
      year: 1999
      artist: Prince
      letter: Z
    classes:
      - "{letter}0"
      - "{letter}1"
      - "{letter}2"
    rm_systems:
      - name: SystemA
        processes: {}
    airlines:
      AL1:
        rm_system: SystemA
      AL2:
        rm_system: SystemA
        classes:
          - "F{year}"
          - "{letter}zz"
          - "{artist}"
    """
    content = yaml.safe_load(io.StringIO(demo))
    loaded = Config.model_validate(content)
    assert loaded.airlines["AL1"].classes == ["Z0", "Z1", "Z2"]
    assert loaded.airlines["AL2"].classes == ["F1999", "Zzz", "Prince"]
