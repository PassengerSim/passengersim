"""Tests for deprecation warnings in choice model fields.

Specifically covers:

* The deprecated ``basefare_mult`` field on ``PodsChoiceModel``, which should
  raise a :class:`DeprecationWarning` whenever a non-``None`` value is supplied.
* The renaming of ``basefare_mult2`` to ``reference_price_multiplier``: configs that use
  the old name should have their value silently migrated; supplying both names at
  the same time should raise a ``ValueError``.
"""

import warnings

import pytest

from passengersim.config.choice_model import PodsChoiceModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pods(**kwargs) -> PodsChoiceModel:
    """Convenience factory that always supplies the required ``kind`` and ``name`` fields."""
    kwargs.setdefault("name", "test")
    return PodsChoiceModel(kind="pods", **kwargs)


# ---------------------------------------------------------------------------
# Tests – basefare_mult deprecation
# ---------------------------------------------------------------------------


class TestBasefareMultDeprecation:
    """Verify that setting ``basefare_mult`` raises ``DeprecationWarning``."""

    def test_setting_value_raises_runtime_warning(self):
        """A non-None value for basefare_mult must trigger a DeprecationWarning."""
        with pytest.warns(RuntimeWarning, match="basefare_mult"):
            _pods(basefare_mult=0.8)

    def test_warning_message_mentions_replacement(self):
        """The warning message should point users to reference_price_multiplier."""
        with pytest.warns(RuntimeWarning, match="reference_price_multiplier"):
            _pods(basefare_mult=1.5)

    def test_no_warning_when_field_omitted(self):
        """Omitting basefare_mult entirely must not produce a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _pods()

        dep = [w for w in caught if issubclass(w.category, DeprecationWarning) and "basefare_mult" in str(w.message)]
        assert not dep, f"Unexpected DeprecationWarning(s): {dep}"

    def test_no_warning_when_field_explicitly_none(self):
        """Passing ``basefare_mult=None`` explicitly must not produce a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _pods(basefare_mult=None)

        dep = [w for w in caught if issubclass(w.category, DeprecationWarning) and "basefare_mult" in str(w.message)]
        assert not dep, f"Unexpected DeprecationWarning(s): {dep}"

    def test_value_is_still_stored(self):
        """Despite being deprecated, the supplied value should be stored on the model."""
        with pytest.warns(RuntimeWarning):
            m = _pods(basefare_mult=0.75)
        assert m.basefare_mult == pytest.approx(0.75)

    def test_zero_value_raises_warning(self):
        """Zero is a valid (non-None) value and should still trigger the warning."""
        with pytest.warns(RuntimeWarning, match="basefare_mult"):
            _pods(basefare_mult=0.0)

    def test_basefare_multiplier_unaffected(self):
        """Setting reference_price_multiplier must never raise a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = _pods(reference_price_multiplier=2.0)

        dep = [w for w in caught if issubclass(w.category, DeprecationWarning) and "basefare_mult" in str(w.message)]
        assert not dep, f"Unexpected DeprecationWarning(s): {dep}"
        assert m.reference_price_multiplier == pytest.approx(2.0)

    def test_warning_category_is_exactly_runtime_warning(self):
        """The warning category must be DeprecationWarning (not a subclass)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _pods(basefare_mult=1.0)

        dep = [w for w in caught if "basefare_mult" in str(w.message)]
        assert dep, "No warning at all was raised for basefare_mult"
        assert dep[0].category is RuntimeWarning


# ---------------------------------------------------------------------------
# Tests – basefare_mult2 → reference_price_multiplier rename
# ---------------------------------------------------------------------------


class TestBasefareMultiplierRename:
    """Verify that the old ``basefare_mult2`` name is silently migrated."""

    def test_new_name_accepted(self):
        """Using ``reference_price_multiplier`` directly should work without any warning or error."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = _pods(reference_price_multiplier=1.5)

        dep = [w for w in caught if issubclass(w.category, (DeprecationWarning, UserWarning))]
        assert not dep, f"Unexpected warning(s): {dep}"
        assert m.reference_price_multiplier == pytest.approx(1.5)

    def test_old_name_silently_migrated(self):
        """Using the old ``basefare_mult2`` name should silently set ``reference_price_multiplier``."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = _pods(basefare_mult2=1.8)

        dep = [w for w in caught if issubclass(w.category, (DeprecationWarning, UserWarning))]
        assert not dep, f"Unexpected warning(s) during silent migration: {dep}"
        assert m.reference_price_multiplier == pytest.approx(1.8)

    def test_old_name_does_not_create_extra_attribute(self):
        """The migrated model must not have a ``basefare_mult2`` attribute."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            m = _pods(basefare_mult2=1.8)

        assert not hasattr(m, "basefare_mult2"), "basefare_mult2 should not exist as an attribute after migration"

    def test_default_value_preserved(self):
        """When neither name is given, the default value (1.0) should be set."""
        m = _pods()
        assert m.reference_price_multiplier == pytest.approx(1.0)

    def test_none_old_name_keeps_default(self):
        """Passing ``basefare_mult2=None`` should leave ``reference_price_multiplier`` at its default."""
        m = _pods(basefare_mult2=None)
        assert m.reference_price_multiplier == pytest.approx(1.0)

    def test_old_name_overridden_by_none_new_name(self):
        """Old name value wins when new name is explicitly None."""
        m = _pods(basefare_mult2=2.5, reference_price_multiplier=None)
        assert m.reference_price_multiplier == pytest.approx(2.5)

    def test_both_names_raises_error(self):
        """Specifying both ``basefare_mult2`` and ``reference_price_multiplier`` must raise ValueError."""
        with pytest.raises(Exception, match="basefare_mult2"):
            _pods(basefare_mult2=1.0, reference_price_multiplier=2.0)

    def test_both_names_error_message_mentions_new_name(self):
        """The conflict error message should mention ``reference_price_multiplier``."""
        with pytest.raises(Exception, match="reference_price_multiplier"):
            _pods(basefare_mult2=1.0, reference_price_multiplier=2.0)
