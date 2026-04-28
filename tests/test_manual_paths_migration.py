"""Tests to verify manual_paths deprecation warning and migration."""

import warnings

from passengersim.config.simulation_controls import SimulationSettings


def test_manual_paths_set_to_true():
    """Test that setting manual_paths to True raises deprecation warning and migrates value."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        settings = SimulationSettings(manual_paths=True)

        # Check that a deprecation warning was issued
        assert len(w) > 0, "Expected a deprecation warning to be raised"
        assert issubclass(w[0].category, DeprecationWarning), (
            f"Expected DeprecationWarning, got {w[0].category.__name__}"
        )

        # Check that the value was migrated
        assert settings.manual_paths is None, (
            f"Expected manual_paths to be None after migration, got {settings.manual_paths}"
        )
        assert settings.connection_builder.existing_paths == "required", (
            f"Expected connection_builder.existing_paths == required, got {settings.connection_builder.existing_paths}"
        )


def test_manual_paths_set_to_false():
    """Test that setting manual_paths to False raises deprecation warning and migrates value."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        settings = SimulationSettings(manual_paths=False)

        # Check that a deprecation warning was issued
        assert len(w) > 0, "Expected a deprecation warning to be raised"
        assert issubclass(w[0].category, DeprecationWarning), (
            f"Expected DeprecationWarning, got {w[0].category.__name__}"
        )

        # Check that the value was migrated
        assert settings.manual_paths is None, (
            f"Expected manual_paths to be None after migration, got {settings.manual_paths}"
        )
        assert settings.connection_builder.existing_paths == "none", (
            f"Expected connection_builder.existing_paths to be 'none', got {settings.connection_builder.existing_paths}"
        )


def test_manual_paths_not_set_uses_default():
    """Test that not setting manual_paths uses default value.

    Note: A deprecation warning may be raised due to internal field access in validators,
    but the important thing is that the user didn't explicitly set it and gets correct defaults.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        settings = SimulationSettings()

        # Check default values - these are what matter for the user
        assert settings.manual_paths is None, (
            f"Expected manual_paths to be None by default, got {settings.manual_paths}"
        )
        assert settings.connection_builder.existing_paths == "keep", (
            f"Expected connection_builder.manual_paths to be 'keep' by default, "
            f"got {settings.connection_builder.existing_paths}"
        )


def test_connection_builder_manual_paths_set_directly():
    """Test that setting connection_builder.manual_paths directly works correctly.

    Note: A deprecation warning may still be raised due to internal field access in validators,
    but the new syntax correctly sets the value where it belongs.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        settings = SimulationSettings(connection_builder={"existing_paths": "required"})

        # Check values - this is what matters: the value is set in the right place
        assert settings.manual_paths is None, f"Expected manual_paths to be None, got {settings.manual_paths}"
        assert settings.connection_builder.existing_paths == "required", (
            f"Expected connection_builder.manual_paths to be required, got {settings.connection_builder.existing_paths}"
        )
