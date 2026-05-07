"""Unit tests for the ToddCurve.probabilities pydantic validator."""

import warnings

import pytest
from pydantic import ValidationError

from passengersim.config.todd_curves import ToddCurve


def test_probabilities_none():
    """None (the default) should pass through unchanged."""
    t = ToddCurve(name="test", probabilities=None)
    assert t.probabilities is None


def test_probabilities_already_normalized_no_warning():
    """A fully-normalized dict (sum == 1.0) should not trigger a warning."""
    probs = {h: 1 / 24 for h in range(24)}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = ToddCurve(name="test", probabilities=probs)
    assert len(w) == 0, f"Unexpected warning(s): {w}"
    assert abs(sum(t.probabilities.values()) - 1.0) < 1e-9


def test_probabilities_rescaling_produces_warning():
    """Probabilities that don't sum to 1.0 should trigger a UserWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ToddCurve(name="test", probabilities={8: 2.0, 12: 3.0, 18: 5.0})
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)


def test_probabilities_rescaled_values_correct():
    """After rescaling, values should be proportionally correct and sum to 1.0."""
    t = ToddCurve(name="test", probabilities={8: 2.0, 12: 3.0, 18: 5.0})
    assert abs(sum(t.probabilities.values()) - 1.0) < 1e-9
    assert abs(t.probabilities[8] - 0.2) < 1e-9
    assert abs(t.probabilities[12] - 0.3) < 1e-9
    assert abs(t.probabilities[18] - 0.5) < 1e-9


def test_probabilities_missing_keys_filled_with_zero():
    """All 24 hours (0–23) must be present; missing ones are implicitly 0.0."""
    t = ToddCurve(name="test", probabilities={8: 2.0, 12: 3.0, 18: 5.0})
    assert len(t.probabilities) == 24
    assert t.probabilities[0] == 0.0
    assert t.probabilities[23] == 0.0


def test_probabilities_out_of_range_key_raises():
    """A key outside 0–23 should raise a ValidationError."""
    with pytest.raises(ValidationError):
        ToddCurve(name="test", probabilities={25: 1.0})

    with pytest.raises(ValidationError):
        ToddCurve(name="test", probabilities={-1: 1.0})


def test_probabilities_negative_value_raises():
    """A negative probability value should raise a ValidationError."""
    with pytest.raises(ValidationError):
        ToddCurve(name="test", probabilities={5: -1.0})


def test_probabilities_all_zero_raises():
    """All-zero probabilities cannot be normalised and should raise a ValidationError."""
    with pytest.raises(ValidationError):
        ToddCurve(name="test", probabilities={0: 0.0, 1: 0.0})


def test_probabilities_string_keys_coerced():
    """String keys (as produced by YAML parsers) should be coerced to int."""
    t = ToddCurve(name="test", probabilities={"8": 0.5, "20": 0.5})
    assert abs(sum(t.probabilities.values()) - 1.0) < 1e-9
    assert abs(t.probabilities[8] - 0.5) < 1e-9
    assert abs(t.probabilities[20] - 0.5) < 1e-9


def test_probabilities_single_hour():
    """A single non-zero hour should normalise to 1.0 for that hour."""
    t = ToddCurve(name="test", probabilities={12: 42.0})
    assert abs(t.probabilities[12] - 1.0) < 1e-9
    assert all(t.probabilities[h] == 0.0 for h in range(24) if h != 12)


def test_probabilities_keys_span_full_24_hour_range():
    """Validator should cover hours 0 through 23 inclusive (edge hours)."""
    t = ToddCurve(name="test", probabilities={0: 1.0, 23: 1.0})
    assert abs(t.probabilities[0] - 0.5) < 1e-9
    assert abs(t.probabilities[23] - 0.5) < 1e-9
