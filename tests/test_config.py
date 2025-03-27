import json

import pytest

from passengersim import demo_network
from passengersim.config import Config, DatabaseConfig, manipulate


@pytest.fixture
def config() -> Config:
    input_file = demo_network("3MKT/08-untrunc-em")
    cfg = Config.from_yaml(input_file)
    cfg.simulation_controls.num_trials = 2
    cfg.simulation_controls.num_samples = 150
    cfg.simulation_controls.burn_samples = 75
    cfg.outputs.reports.clear()
    cfg.outputs.reports.add("*")
    return cfg


def test_db_serialization(config: Config):
    z = config.model_dump()
    assert isinstance(z, dict)
    assert "db" in z
    assert isinstance(z["db"], dict)
    assert sorted(z["db"].keys()) == [
        "commit_count_delay",
        "dcp_write_hooks",
        "engine",
        "fast",
        "filename",
        "pragmas",
        "store_displacements",
        "store_leg_bid_prices",
        "write_items",
    ]
    assert z["db"]["engine"] == "sqlite"

    # when explicitly setting the database to None in the Config object,
    # the value of the 'db' key in the object stays as None and is not
    # converted to a null database, and also the serialized form
    # should be None, not a null database
    config.db = None
    assert config.db is None
    z = config.model_dump()
    assert isinstance(z, dict)
    assert "db" in z
    assert z["db"] is None

    # when loading the serialized form, the db attribute should get
    # loaded as a null database
    cfg0 = Config.model_validate(z)
    assert isinstance(cfg0.db, DatabaseConfig)
    assert cfg0.db.filename is None
    assert not config.find_differences(cfg0)

    # test round trip through json, in the serialized form
    # the 'db' key should a None, but when deserialized
    # the database should end up as a null database
    j = cfg0.model_dump_json()
    jd = json.loads(j)
    assert jd["db"] is None
    cfg1 = Config.model_validate(jd)
    assert isinstance(cfg1.db, DatabaseConfig)
    assert cfg1.db.filename is None

    assert not config.find_differences(cfg1)


def test_output_serialization(config):
    z = config.model_dump()
    assert isinstance(z, dict)
    assert "outputs" in z
    assert isinstance(z["outputs"], dict)
    assert sorted(z["outputs"].keys()) == [
        "disk",
        "excel",
        "html",
        "log_reports",
        "pickle",
        "reports",
    ]
    assert config.outputs.reports == {"*"}
    assert isinstance(z["outputs"]["reports"], list)
    assert z["outputs"]["reports"] == ["*"]


def test_restriction_stripping(config):
    cfg0 = manipulate.strip_all_restrictions(config)
    assert not cfg0.choice_models.business.restrictions
    assert not cfg0.choice_models.leisure.restrictions
    assert config.choice_models.business.restrictions
    assert config.choice_models.leisure.restrictions
    for fare in cfg0.fares:
        assert not fare.restrictions

    cfg1 = manipulate.strip_all_restrictions(config, inplace=True)
    assert not cfg1.choice_models.business.restrictions
    assert not cfg1.choice_models.leisure.restrictions
    assert not config.choice_models.business.restrictions
    assert not config.choice_models.leisure.restrictions
    for fare in cfg1.fares:
        assert not fare.restrictions
    for fare in config.fares:
        assert not fare.restrictions
