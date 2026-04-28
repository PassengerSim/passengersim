from pydantic import ValidationError
from pytest import raises

from passengersim.config.carriers import Carrier


def test_carrier_rm_inputs_def_only():
    c = Carrier(name="ACME", rm_system_options={"name": "RMX"})
    assert c.name == "ACME"
    assert c.rm_system == "RMX"


def test_carrier_rm_inputs_name_and_def():
    c2 = Carrier(name="ACME2", rm_system="EXPLICIT", rm_system_options={"name": "EXPLICIT"})
    assert c2.name == "ACME2"
    assert c2.rm_system == "EXPLICIT"


def test_carrier_rm_inputs_name_dict():
    c2 = Carrier(name="ACME2", rm_system={"name": "EXPLICIT", "param": 42})
    assert c2.name == "ACME2"
    assert c2.rm_system == "EXPLICIT"
    assert c2.rm_system_options == {"param": 42}


def test_carrier_rm_inputs_dict_conflict():
    with raises(ValidationError):
        _ = Carrier(
            name="ACME2",
            rm_system={"name": "EXPLICIT", "param": 42},
            rm_system_options={"name": "EXPLICIT", "param": 73},
        )


def test_carrier_missing_rm():
    with raises(ValidationError):
        _ = Carrier(name="ACME3")


def test_carrier_rm_name_conflict():
    with raises(ValidationError):
        # fails on mismatched rm_system and rm_system_options name
        _ = Carrier(name="ACME4", rm_system="EXPLICIT", rm_system_options={"name": "RMX"})
