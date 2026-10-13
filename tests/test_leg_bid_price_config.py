"""Configuration contract for optional native all-leg bid-price collection."""

import pytest
import yaml
from pydantic import ValidationError

from passengersim.config import Config
from passengersim.config.simulation_controls import SimulationSettings


def test_collection_defaults_on() -> None:
    assert SimulationSettings().collect_leg_bid_prices is True
    config = Config.model_validate({"db": None})
    assert config.simulation_controls.collect_leg_bid_prices is True


@pytest.mark.parametrize("enabled", [False, True])
def test_collection_config_round_trip(enabled: bool) -> None:
    config = Config.model_validate(
        yaml.safe_load(f"db: null\nsimulation_controls:\n  collect_leg_bid_prices: {str(enabled).lower()}\n")
    )
    restored = Config.model_validate_json(config.model_dump_json())
    assert restored.db is None
    assert restored.simulation_controls.collect_leg_bid_prices is enabled


def test_collection_rejects_leg_subsets() -> None:
    settings = SimulationSettings()
    with pytest.raises(ValidationError):
        settings.collect_leg_bid_prices = [101, 102]
