import pytest

from passengersim import Config, demo_network
from passengersim.transforms import clean_restrictions


@pytest.fixture
def mixed_case_config() -> Config:
    """Create a config whose matching restriction names use different cases."""
    cfg = Config.from_yaml(demo_network("3MKT/08-untrunc-em"))

    for choice_model in cfg.choice_models.values():
        if choice_model.restrictions:
            choice_model.restrictions = {name.upper(): value for name, value in choice_model.restrictions.items()}

    for fare in cfg.fares:
        fare.restrictions = [name.lower() for name in fare.restrictions]

    choice_model = next(cm for cm in cfg.choice_models.values() if cm.restrictions)
    choice_model.restrictions["CHOICE_ONLY"] = 1.0
    cfg.fares[0].restrictions.extend(["fare_only", "FARE_ONLY"])
    return cfg


def test_clean_restrictions_matches_case_insensitively(mixed_case_config: Config):
    cleaned = clean_restrictions(mixed_case_config)

    fare_names = {name for fare in cleaned.fares for name in fare.restrictions}
    choice_model_names = {
        name for choice_model in cleaned.choice_models.values() for name in (choice_model.restrictions or {})
    }

    assert {name.casefold() for name in fare_names} == {name.casefold() for name in choice_model_names}
    assert all(name == name.lower() for name in fare_names)
    assert all(name == name.upper() for name in choice_model_names)
    assert "fare_only" not in {name.casefold() for name in fare_names}
    assert "choice_only" not in {name.casefold() for name in choice_model_names}

    # The default behavior returns a copy and leaves the source untouched.
    assert "CHOICE_ONLY" in {
        name for choice_model in mixed_case_config.choice_models.values() for name in (choice_model.restrictions or {})
    }
    assert any(name.casefold() == "fare_only" for name in mixed_case_config.fares[0].restrictions)


def test_clean_restrictions_case_insensitive_inplace(mixed_case_config: Config):
    cleaned = clean_restrictions(mixed_case_config, inplace=True)

    assert cleaned is mixed_case_config
    assert all(
        name.casefold() != "choice_only"
        for choice_model in cleaned.choice_models.values()
        for name in (choice_model.restrictions or {})
    )
    assert all(name.casefold() != "fare_only" for fare in cleaned.fares for name in fare.restrictions)
