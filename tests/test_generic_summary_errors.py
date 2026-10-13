from types import SimpleNamespace

import pandas as pd
import pytest

from passengersim.summaries.generic import GenericSimulationTables


@pytest.mark.parametrize("chained", [False, True])
def test_extraction_warning_includes_traceback_and_continues(monkeypatch, chained):
    """Keep debugging details from failed extractors without losing other tables."""

    def fail_in_helper():
        raise ValueError("invalid summary data")

    def failing_extractor(sim):
        if chained:
            try:
                fail_in_helper()
            except ValueError as err:
                raise RuntimeError("extraction failed") from err
        fail_in_helper()

    expected = pd.DataFrame({"value": [42]})
    monkeypatch.setattr(
        GenericSimulationTables,
        "_std_extract",
        {"broken": failing_extractor, "working": lambda sim: expected},
    )
    sim = SimpleNamespace(
        eng=SimpleNamespace(num_trials_completed=2, num_samples=10, burn_samples=2),
        config=None,
        db=None,
        callback_data=None,
    )

    with pytest.warns(UserWarning, match="in broken:") as caught:
        summary = GenericSimulationTables.extract(sim)

    assert len(caught) == 1
    message = str(caught[0].message)
    assert "Traceback (most recent call last):" in message
    assert 'File "' in message
    assert ", line " in message
    assert "in failing_extractor" in message
    assert "in fail_in_helper" in message
    assert "ValueError: invalid summary data" in message
    if chained:
        assert "The above exception was the direct cause" in message
        assert "RuntimeError: extraction failed" in message
    assert caught[0].filename == __file__
    assert "broken" not in summary._data
    pd.testing.assert_frame_equal(summary._data["working"], expected)
    assert summary.n_total_samples == 16
