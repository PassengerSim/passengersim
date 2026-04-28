import warnings

import numpy as np
import pandas as pd

__all__ = ["beeswarm"]


def create_hex_centers(
    n_points: int,
    aspect_ratio: float = 1.0,
    *,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Generates the center points for a pointy-top hexagonal grid.
    """

    # number of rows of hexagons
    ny = int(np.round(np.sqrt(0.8660254038 * n_points / aspect_ratio), 0))

    # number of columns of hexagons
    nx = int(np.round(n_points / ny, 0))

    # circumradius of the hexagons - this controls the spacing of the points.
    # We can adjust this to get a more or less dense grid, but since we are
    # normalizing the coordinates to [0, 1], the absolute value doesn't matter,
    # we just need it to be consistent.
    radius = 1.0

    # Calculate key dimensions
    width = 2 * radius
    height = np.sqrt(3) * radius

    # Create a base rectangular grid
    x = np.arange(nx, dtype=float) * width * 0.75
    y = np.arange(ny, dtype=float) * height

    # Use meshgrid to get 2D arrays of coordinates
    xv, yv = np.meshgrid(x, y)

    # Offset every other row
    xv[1::2, :] += width * 0.75 / 2.0

    if normalize:
        norm = np.maximum(yv.max(), 1)
        yv /= norm  # Normalize y to [0, 1]
        xv /= norm  # Normalize x to preserve aspect ratio

    return pd.DataFrame({"x": xv.flatten() / aspect_ratio, "y": yv.flatten()})


def _hex_alignment(points: pd.DataFrame, x: str, y: str) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to compute common normalization parameters for matching points to hex centers.

    This is done without actually doing the matching, so that faceted matching can
    be done using a common scaling."""

    combo_xy = points[[x, y]].to_numpy()

    # normalize combo_xy to [0.05, 0.95]
    xy_min = combo_xy.min(axis=0)
    xy_max = combo_xy.max(axis=0)
    xy_range = xy_max - xy_min
    xy_range[xy_range == 0] = 1.0  # Avoid division by zero for constant columns
    if np.any(xy_max - xy_min == 0):
        raise ValueError("Cannot normalize points with zero range in x or y")

    return xy_min, xy_range


def _match_points_to_hex_centers(
    points: pd.DataFrame, hex_centers: pd.DataFrame, x: str, y: str, xy_min: np.ndarray, xy_range: np.ndarray
) -> pd.DataFrame:
    # Greedy one-to-one nearest match: each hex_centers point can be used only once
    hex_xy = hex_centers[["x", "y"]].to_numpy()
    combo_xy = points[[x, y]].to_numpy()

    combo_xy = (combo_xy - xy_min) / xy_range
    combo_xy = combo_xy * 0.9 + 0.05

    available = np.ones(len(hex_centers), dtype=bool)
    chosen_idx = np.full(len(points), -1, dtype=int)
    chosen_dist = np.empty(len(points), dtype=float)

    for i, p in enumerate(combo_xy):
        avail_idx = np.flatnonzero(available)
        d2 = np.sum((hex_xy[avail_idx] - p) ** 2, axis=1)
        best_local = np.argmin(d2)
        best_idx = avail_idx[best_local]

        chosen_idx[i] = best_idx
        chosen_dist[i] = np.sqrt(d2[best_local])
        available[best_idx] = False

    matched_df = points.copy()
    matched_df["hex_index"] = chosen_idx
    matched_df[f"original_{x}"] = points[x]
    matched_df[f"original_{y}"] = points[y]

    # reverse the normalization to get the hex centers back to the original scale
    hex_centers["x_"] = ((hex_centers["x"] - 0.05) / 0.9) * xy_range[0] + xy_min[0]
    hex_centers["y_"] = ((hex_centers["y"] - 0.05) / 0.9) * xy_range[1] + xy_min[1]

    matched_df[[x, y]] = hex_centers.loc[chosen_idx, ["x_", "y_"]].to_numpy()
    matched_df["_hex_distance_"] = chosen_dist

    return matched_df


def beeswarm(
    points: pd.DataFrame, x: str, y: str, aspect_ratio: float = 1.25, n_hex: int = 10_000, facet: str | None = None
) -> pd.DataFrame:
    """
    Transforms the input points into a beeswarm layout by matching them to a hexagonal grid.

    This is useful for visualizing distributions of points without overlap, while preserving
    the overall shape of the data. The function generates a hexagonal grid of points and then
    matches each input point to the nearest available hexagonal center, ensuring that no two
    input points are assigned to the same hexagonal center.

    Parameters
    ----------
    points : pd.DataFrame
        The input DataFrame containing the points to be transformed.
    x : str
        The name of the column in `points` that contains the x-coordinates.
    y : str
        The name of the column in `points` that contains the y-coordinates.
    aspect_ratio : float, optional
        The aspect ratio (width/height) of the hexagonal grid. Default is 1.25.
    n_hex : int, optional
        The number of hexagonal centers to generate for the grid. Default is 10,000
        which should be sufficient for most datasets, but can be increased for
        larger datasets to improve the beeswarm layout.
    facet : str, optional
        The name of the column in `points` to facet by. If provided, the function
        will generate one common hexagonal grid, and allow one hex point match per
        row within each facet instead of across the whole dataset.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with the original points matched to hexagonal grid centers,
        including original coordinates and distance to hex centers.
    """
    if n_hex < len(points) * 10:
        warnings.warn(
            f"Data has {len(points)} rows. "
            f"`n_hex` is set to {n_hex} which may be too low for this data, "
            "which could lead to a misleading beeswarm layout. Consider "
            "increasing `n_hex` for better results.",
            stacklevel=2,
        )
    hex_centers = create_hex_centers(n_hex, aspect_ratio=aspect_ratio)
    xy_min, xy_range = _hex_alignment(points, x, y)
    if facet is None:
        matched_df = _match_points_to_hex_centers(points, hex_centers, x, y, xy_min, xy_range)
    else:
        matched_queue = []
        for facet_value, group in points.groupby(facet):
            matched_group = _match_points_to_hex_centers(group, hex_centers, x, y, xy_min, xy_range)
            matched_group[facet] = facet_value
            matched_queue.append(matched_group)
        matched_df = pd.concat(matched_queue, ignore_index=True)
    return matched_df
