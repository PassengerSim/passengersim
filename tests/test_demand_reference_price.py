"""Tests for the Demand config class rename of ``reference_fare`` to
``reference_price``, including the backward-compatibility shim that accepts
the legacy name at validation time and at attribute-access time."""

from __future__ import annotations

import warnings

import pytest

from passengersim.config.demands import Demand


def _make_demand(**overrides):
    """Helper to construct a Demand with minimal required fields."""
    kwargs = dict(orig="A", dest="B", segment="biz", base_demand=1.0)
    kwargs.update(overrides)
    return Demand(**kwargs)


def test_reference_price_is_primary_name():
    """The new field should be accepted and readable without warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        d = _make_demand(reference_price=100.0)
        assert d.reference_price == 100.0


def test_legacy_reference_fare_input_is_migrated():
    """Providing the legacy ``reference_fare`` keyword should move the value
    to ``reference_price`` and emit a DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="reference_price"):
        d = _make_demand(reference_fare=250.0)
    assert d.reference_price == 250.0


def test_legacy_reference_fare_attribute_access():
    """Reading ``reference_fare`` should return ``reference_price`` and
    emit a DeprecationWarning."""
    d = _make_demand(reference_price=75.0)
    with pytest.warns(DeprecationWarning, match="reference_price"):
        value = d.reference_fare
    assert value == 75.0


def test_legacy_reference_fare_attribute_set():
    """Assigning ``reference_fare`` should update ``reference_price`` and
    emit a DeprecationWarning."""
    d = _make_demand(reference_price=75.0)
    with pytest.warns(DeprecationWarning, match="reference_price"):
        d.reference_fare = 555.0
    assert d.reference_price == 555.0


def test_both_names_supplied_prefers_reference_price():
    """If both are supplied, ``reference_price`` takes precedence and
    ``reference_fare`` is discarded with a warning."""
    with pytest.warns(DeprecationWarning):
        d = _make_demand(reference_price=10.0, reference_fare=99.0)
    assert d.reference_price == 10.0


def test_no_warning_on_normal_usage():
    """Using only ``reference_price`` should not emit any DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        d = _make_demand(reference_price=42.0)
        # read and write the new name, no warnings expected
        assert d.reference_price == 42.0
        d.reference_price = 43.0
        assert d.reference_price == 43.0
