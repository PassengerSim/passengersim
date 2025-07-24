import pytest

from passengersim import Config, demo_network
from passengersim.config.fares import Fare
from passengersim.transforms.booking_classes import class_rename, pad_digits


def test_basic_padding():
    assert pad_digits(["A1", "B2", "C33"]) == {"A1": "A01", "B2": "B02", "C33": "C33"}


def test_no_padding_needed():
    assert pad_digits(["X10", "Y20", "Z30"]) == {"X10": "X10", "Y20": "Y20", "Z30": "Z30"}


def test_mixed_non_matching():
    assert pad_digits(["A1", "B2", "C"]) == {"A1": "A1", "B2": "B2", "C": "C"}


def test_all_non_matching():
    assert pad_digits(["foo", "bar", "baz"]) == {"foo": "foo", "bar": "bar", "baz": "baz"}


def test_empty_list():
    assert pad_digits([]) == {}


def test_leading_zeros():
    assert pad_digits(["A01", "B2", "C003"]) == {"A01": "A001", "B2": "B002", "C003": "C003"}


@pytest.fixture
def config() -> Config:
    input_file = demo_network("3MKT/08-untrunc-em")
    cfg = Config.from_yaml(input_file)
    return cfg


def test_class_rename(config: Config):
    # add some two digit booking classes
    config.fares.append(
        Fare(carrier="AL1", orig="BOS", dest="ORD", booking_class="Y12", price=5.0, advance_purchase=35)
    )
    print(config.fares)
    cfg = config.model_revalidate()

    # check that booking classes are not yet padded correctly
    for f in cfg.fares:
        assert f.booking_class in ["Y0", "Y1", "Y2", "Y3", "Y4", "Y5", "Y12"]

    cfg = class_rename(cfg)

    # check that booking classes are now padded correctly
    for f in cfg.fares:
        assert f.booking_class in ["Y00", "Y01", "Y02", "Y03", "Y04", "Y05", "Y12"]
