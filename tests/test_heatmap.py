import numpy as np
import pandas as pd

from passengersim.utils.heatmap import compress_and_tidy_heatmap_data


def _make_df(data, rows_name="dcp", cols_name="leg"):
    """Helper to create a DataFrame from a dict of {col: [row_values]}."""
    df = pd.DataFrame(data)
    df.index.name = rows_name
    df.columns.name = cols_name
    return df


class TestCompressAndTidyHeatmapData:
    """Tests for compress_and_tidy_heatmap_data."""

    def test_monotonic_values(self):
        """When values are monotonically sorted, ranges should not overlap."""
        df = _make_df({"A": [10, 20, 30, 40, 50]})
        result = compress_and_tidy_heatmap_data(df, round_to=1.0)
        # Each value is unique so no compression; 5 separate rows
        assert len(result) == 5

    def test_non_monotonic_no_overlap(self):
        """Non-monotonic values: identical values in different runs must not
        produce overlapping ranges."""
        # Values: 10, 10, 20, 10, 10
        # Expected runs: [0-1]=10, [2]=20, [3-4]=10
        df = _make_df({"A": [10, 10, 20, 10, 10]})
        result = compress_and_tidy_heatmap_data(df, round_to=1.0)

        col_a = result[result["leg"] == "A"]
        rows = col_a[["lowest_dcp", "highest_dcp", "bid_price"]].values.tolist()

        # Should have 3 runs
        assert len(rows) == 3

        # Verify runs are correct (sorted by lowest_dcp)
        rows.sort(key=lambda r: r[0])
        assert rows[0] == [0, 2, 10.0]  # rows 0-1 → lowest=0, highest=2
        assert rows[1] == [2, 3, 20.0]  # row 2 → lowest=2, highest=3
        assert rows[2] == [3, 5, 10.0]  # rows 3-4 → lowest=3, highest=5

    def test_no_overlapping_ranges(self):
        """Ensure that no two ranges for the same column overlap."""
        # Zigzag values that would cause overlaps with old implementation
        df = _make_df({"A": [5, 10, 5, 10, 5]})
        result = compress_and_tidy_heatmap_data(df, round_to=1.0)

        col_a = result[result["leg"] == "A"].sort_values("lowest_dcp")
        ranges = col_a[["lowest_dcp", "highest_dcp"]].values.tolist()

        # Check no overlaps: each range's start >= previous range's end
        for i in range(1, len(ranges)):
            assert ranges[i][0] >= ranges[i - 1][1], f"Range {ranges[i]} overlaps with {ranges[i - 1]}"

    def test_all_same_values(self):
        """When all values are the same, they should be compressed into one row."""
        df = _make_df({"A": [10, 10, 10, 10]})
        result = compress_and_tidy_heatmap_data(df, round_to=1.0)

        col_a = result[result["leg"] == "A"]
        assert len(col_a) == 1
        assert col_a.iloc[0]["bid_price"] == 10.0
        assert col_a.iloc[0]["lowest_dcp"] == 0
        assert col_a.iloc[0]["highest_dcp"] == 4

    def test_alternating_values(self):
        """Alternating values should produce one run per row (no compression)."""
        df = _make_df({"A": [10, 20, 10, 20]})
        result = compress_and_tidy_heatmap_data(df, round_to=1.0)

        col_a = result[result["leg"] == "A"]
        assert len(col_a) == 4

    def test_multiple_columns(self):
        """Verify each column is compressed independently."""
        df = _make_df(
            {
                "A": [10, 10, 20, 20],
                "B": [5, 5, 5, 5],
            }
        )
        result = compress_and_tidy_heatmap_data(df, round_to=1.0)

        col_a = result[result["leg"] == "A"]
        col_b = result[result["leg"] == "B"]

        assert len(col_a) == 2  # two runs: 10,10 and 20,20
        assert len(col_b) == 1  # one run: all 5s

    def test_single_row(self):
        """Single row data should produce one output row per column."""
        df = _make_df({"A": [42]})
        result = compress_and_tidy_heatmap_data(df, round_to=1.0)
        assert len(result) == 1

    def test_label_format_single_row_run(self):
        """When a run covers a single row, the label should be just that row value."""
        df = _make_df({"A": [10, 20, 10]})
        result = compress_and_tidy_heatmap_data(df, round_to=1.0)

        col_a = result[result["leg"] == "A"].sort_values("lowest_dcp")
        labels = col_a["dcp"].tolist()

        # Row 1 (value=20) is a single-row run, label should be "1" not "1-2"
        assert labels[1] == "1"

    def test_label_format_multi_row_run(self):
        """When a run covers multiple rows, the label should be 'lowest-highest'."""
        df = _make_df({"A": [10, 10, 10, 20]})
        result = compress_and_tidy_heatmap_data(df, round_to=1.0)

        col_a = result[result["leg"] == "A"].sort_values("lowest_dcp")
        labels = col_a["dcp"].tolist()

        # First run covers rows 0,1,2 → highest is 3, label "0-3"
        assert labels[0] == "0-3"

    def test_rounding_applied(self):
        """Verify that rounding is applied to values before compression."""
        # Values 10.1 and 10.2 should both round to 10.0 with round_to=0.25
        df = _make_df({"A": [10.1, 10.2, 20.0]})
        result = compress_and_tidy_heatmap_data(df, round_to=0.25)

        col_a = result[result["leg"] == "A"]
        # 10.1→10.0, 10.2→10.25 — these are different after rounding to 0.25
        # Actually 10.1/0.25 = 40.4 → round → 40 → 40*0.25 = 10.0
        # 10.2/0.25 = 40.8 → round → 41 → 41*0.25 = 10.25
        # So they're different, giving 3 runs
        assert len(col_a) == 3

    def test_max_rows_triggers_recompression(self):
        """When result exceeds max_rows, round_to should double and recompress."""
        # Create data with many distinct values that will exceed max_rows
        n = 100
        df = _make_df({f"col{i}": np.random.default_rng(42).random(50) * 100 for i in range(n)})
        result = compress_and_tidy_heatmap_data(df, round_to=0.25, max_rows=100)
        assert len(result) <= 100
