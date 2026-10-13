"""SQL positional bindings preserve the scenario and burn-period filters."""

from contextlib import closing

import pytest

from passengersim.database import Database
from passengersim.database.common_queries import fare_class_mix


@pytest.mark.filterwarnings("error:Binding .* is a named parameter:DeprecationWarning")
def test_fare_class_mix_parameter_order() -> None:
    """Distinct scenarios and burned samples must not contaminate the averages."""
    with closing(Database(filename=":memory:")) as db:
        db.execute("CREATE TABLE fare_summary (trial, sample, scenario, carrier, booking_class, sold, revenue)")
        for row in (
            (0, 0, "chosen", "AL1", "Y", 100, 10000),
            (0, 2, "chosen", "AL1", "Y", 2, 200),
            (0, 3, "chosen", "AL1", "Y", 4, 400),
            (0, 2, "other", "AL1", "Y", 200, 20000),
        ):
            db.execute("INSERT INTO fare_summary VALUES (?, ?, ?, ?, ?, ?, ?)", row)

        result = fare_class_mix(db, scenario="chosen", burn_samples=2)

        assert result.index.tolist() == [("AL1", "Y")]
        assert result.loc[("AL1", "Y")].to_dict() == {
            "avg_sold": 3.0,
            "avg_revenue": 300.0,
            "avg_price": 100.0,
        }
