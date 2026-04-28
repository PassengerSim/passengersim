import numpy as np
import pandas as pd


def round_to_nearest(values, nearest: float = 1.0):
    return np.round(values / nearest, 0) * nearest


def compress_and_tidy_heatmap_data(
    df: pd.DataFrame, value_name: str = "bid_price", round_to: float = 0.25, max_rows: int = 4500
):
    """Partially aggregate heatmap data so that there are not too many data rows.

    Altair performs better when the size of the underlying data is limited.
    """
    cols_name = df.columns.name
    rows_name = df.index.name

    tidy = round_to_nearest(df, round_to).melt(ignore_index=False, value_name=value_name).reset_index()

    def _compress(group_data, thing):
        sorted_data = group_data.sort_values(thing)
        # Identify sequential runs of identical values
        run_id = (sorted_data[value_name] != sorted_data[value_name].shift()).cumsum()
        out = sorted_data.groupby(run_id, sort=False).agg(
            **{
                value_name: (value_name, "first"),
                f"lowest_{thing}": (thing, "min"),
                f"highest_{thing}": (thing, "max"),
            },
        )
        out[f"highest_{thing}"] = out[f"highest_{thing}"] + 1
        out = out.set_index([f"lowest_{thing}", f"highest_{thing}"])
        return out

    compressed = tidy.groupby(cols_name).apply(lambda x: _compress(x, rows_name), include_groups=False).reset_index()
    compressed[rows_name] = (
        compressed[f"lowest_{rows_name}"].astype(str) + "-" + compressed[f"highest_{rows_name}"].astype(str)
    )
    compressed[rows_name] = compressed[rows_name].where(
        compressed[f"lowest_{rows_name}"] != compressed[f"highest_{rows_name}"] - 1,
        compressed[f"lowest_{rows_name}"].astype(str),
    )
    if len(compressed) > max_rows:
        return compress_and_tidy_heatmap_data(df, value_name, round_to * 2, max_rows)
    return compressed
