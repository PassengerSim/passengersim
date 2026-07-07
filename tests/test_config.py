import json

import pytest

from passengersim import demo_network
from passengersim.config import Carrier, Config, DatabaseConfig, manipulate


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
        "base_dir",
        "disk",
        "excel",
        "filename_stem",
        "html",
        "log_reports",
        "pickle",
        "reports",
        "sim_state",
    ]
    assert config.outputs.reports == {"*"}
    assert isinstance(z["outputs"]["reports"], list)
    assert z["outputs"]["reports"] == ["*"]


def test_restriction_stripping(config):
    cfg0 = manipulate.strip_fare_restrictions(config)
    assert not cfg0.choice_models.business.restrictions
    assert not cfg0.choice_models.leisure.restrictions
    assert config.choice_models.business.restrictions
    assert config.choice_models.leisure.restrictions
    for fare in cfg0.fares:
        assert not fare.restrictions

    cfg1 = manipulate.strip_fare_restrictions(config, inplace=True)
    assert not cfg1.choice_models.business.restrictions
    assert not cfg1.choice_models.leisure.restrictions
    assert not config.choice_models.business.restrictions
    assert not config.choice_models.leisure.restrictions
    for fare in cfg1.fares:
        assert not fare.restrictions
    for fare in config.fares:
        assert not fare.restrictions


def test_reference_price_variation_vs_multiplier(config: Config):
    """Validate the new check that prevents combining per-OD reference_price
    variation with a non-trivial choice-model reference_price_multiplier.
    """
    # Identify a PODS-style choice model in the fixture to tweak multipliers.
    cm_name, cm = next(iter(config.choice_models.items()))
    assert hasattr(cm, "reference_price_multiplier")

    # Baseline: the fixture should validate as-is (any one or zero conditions OK).
    cfg_ok = Config.model_validate(config.model_dump())
    assert cfg_ok is not None

    # Condition 1 only: introduce reference_price variation within an OD.
    data = config.model_dump()
    # Find at least two demands sharing the same (orig, dest).
    od_groups: dict[tuple[str, str], list[int]] = {}
    for i, d in enumerate(data["demands"]):
        od_groups.setdefault((d["orig"], d["dest"]), []).append(i)
    shared_od = next((idxs for idxs in od_groups.values() if len(idxs) >= 2), None)
    assert shared_od is not None, "fixture must have at least two demands in one OD"
    data["demands"][shared_od[0]]["reference_price"] = 100.0
    data["demands"][shared_od[1]]["reference_price"] = 200.0
    # Ensure all choice models have a neutral (1.0) multiplier.
    for cm_data in data["choice_models"].values():
        if "reference_price_multiplier" in cm_data:
            cm_data["reference_price_multiplier"] = 1.0
    cfg1 = Config.model_validate(data)
    assert cfg1 is not None  # only condition 1 true: OK

    # Condition 2 only: set a non-trivial multiplier, no OD variation.
    data2 = config.model_dump()
    # Normalize: make reference_price constant within every OD group.
    od_first_price: dict[tuple[str, str], float | None] = {}
    for d in data2["demands"]:
        key = (d["orig"], d["dest"])
        od_first_price.setdefault(key, d.get("reference_price"))
        d["reference_price"] = od_first_price[key]
    data2["choice_models"][cm_name]["reference_price_multiplier"] = 1.5
    cfg2 = Config.model_validate(data2)
    assert cfg2 is not None  # only condition 2 true: OK

    # Both conditions true -> should raise.
    data3 = config.model_dump()
    data3["demands"][shared_od[0]]["reference_price"] = 100.0
    data3["demands"][shared_od[1]]["reference_price"] = 200.0
    data3["choice_models"][cm_name]["reference_price_multiplier"] = 1.5
    with pytest.raises(ValueError, match="reference_price_multiplier"):
        Config.model_validate(data3)

    # A None multiplier should NOT count as non-trivial, even with OD variation.
    data4 = config.model_dump()
    data4["demands"][shared_od[0]]["reference_price"] = 100.0
    data4["demands"][shared_od[1]]["reference_price"] = 200.0
    data4["choice_models"][cm_name]["reference_price_multiplier"] = None
    for name, cm_data in data4["choice_models"].items():
        if name == cm_name:
            continue
        if "reference_price_multiplier" in cm_data:
            cm_data["reference_price_multiplier"] = 1.0
    cfg4 = Config.model_validate(data4)
    assert cfg4 is not None


def test_ap_restriction_stripping(config):
    cfg0 = manipulate.strip_ap_restrictions(config)
    for fare in cfg0.fares:
        assert fare.advance_purchase == 0
    assert any(f.advance_purchase > 0 for f in config.fares)

    cfg1 = manipulate.strip_ap_restrictions(config, inplace=True)
    for fare in cfg1.fares:
        assert fare.advance_purchase == 0
    for fare in config.fares:
        assert fare.advance_purchase == 0


def test_cabin_codes():
    # OK
    _c = Carrier(name="AL1", cabin_ordering=["F", "Y"], rm_system="E", classes=["F0", "F1", "Y2", "Y3", "Y4"])


def test_cabin_codes_not_unique():
    with pytest.raises(ValueError, match="cabin codes must be unique"):
        Carrier(name="AL1", cabin_ordering=["Y", "Y"], rm_system="E")


def test_cabin_codes_too_short():
    with pytest.raises(ValueError, match="should have at least 1 character"):
        Carrier(name="AL1", cabin_ordering=["", "Y"], rm_system="E")


def test_cabin_codes_too_long():
    with pytest.raises(ValueError, match="should have at most 1 character"):
        Carrier(name="AL1", cabin_ordering=["ZZZ", "Y"], rm_system="E")


def test_cabin_codes_missing():
    with pytest.raises(ValueError, match="must have at least one cabin code"):
        Carrier(name="AL1", cabin_ordering=[], rm_system="E")


def test_cabin_codes_not_string():
    with pytest.raises(ValueError, match="should be a valid string"):
        Carrier(name="AL1", cabin_ordering=[1, 2, 3], rm_system="E")


def test_class_codes_not_unique():
    with pytest.raises(ValueError, match="class codes must be unique"):
        _c = Carrier(
            name="AL1",
            cabin_ordering=["F", "Y"],
            rm_system="E",
            classes=[("F0", "F"), ("F1", "F"), ("F0", "Y"), ("Y3", "Y"), ("Y4", "Y")],
        )


def test_class_codes_missing_cabin():
    with pytest.raises(ValueError, match="cabin code F not found in cabin ordering"):
        _c = Carrier(
            name="AL1",
            cabin_ordering=["Y"],
            rm_system="E",
            classes=[("F0", "F"), ("F1", "F"), ("Y2", "Y"), ("Y3", "Y"), ("Y4", "Y")],
        )

    with pytest.raises(ValueError, match="class codes must begin with a cabin code character"):
        _c = Carrier(name="AL1", cabin_ordering=["Y"], rm_system="E", classes=["F0", "F1", "Y2", "Y3", "Y4"])
