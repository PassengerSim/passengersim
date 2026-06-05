import warnings

import numpy as np
import pandas as pd

__all__ = ["beeswarm", "bubble_centers"]


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


def _resolve_overlaps(
    cx: np.ndarray,
    cy: np.ndarray,
    w: np.ndarray,
    min_sep: np.ndarray,
    iteration: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Push overlapping circles apart to satisfy ``min_sep`` constraints.

    Displacement is split between each overlapping pair in inverse proportion
    to the circles' weights (``radius²``), so heavier circles move less.

    Parameters
    ----------
    cx, cy : np.ndarray
        Current center x/y coordinates, shape ``(n,)``.
    w : np.ndarray
        Per-circle weights (``radius²``), shape ``(n,)``.
    min_sep : np.ndarray
        Required minimum center-to-center distance for every pair, shape
        ``(n, n)``.
    iteration : int
        Current iteration index used as an RNG seed when circles are
        coincident, ensuring reproducible but varied directions.

    Returns
    -------
    cx, cy : np.ndarray
        Updated center coordinates.
    max_overlap : float
        Maximum constraint violation *before* this step was applied.
    """
    # Pairwise displacement vectors: dxij[i, j] = cx[j] - cx[i]
    dxij = cx[None, :] - cx[:, None]  # (n, n)
    dyij = cy[None, :] - cy[:, None]
    dist = np.sqrt(dxij**2 + dyij**2)

    # Exclude self-pairs.
    np.fill_diagonal(dist, np.inf)

    # Amount by which each pair violates the minimum separation constraint.
    overlap = np.maximum(0.0, min_sep - dist)  # (n, n), symmetric
    max_overlap = float(np.max(overlap))

    if max_overlap == 0.0:
        return cx, cy, 0.0

    # Unit vectors pointing from i toward j; safe floor avoids divide-by-zero.
    safe_dist = np.maximum(dist, 1e-10)
    ux = dxij / safe_dist
    uy = dyij / safe_dist

    # For truly coincident circles, assign reproducible random directions so
    # the algorithm can break the degeneracy and make progress.
    coincident_mask = dist <= 1e-10
    if np.any(coincident_mask):
        rng = np.random.default_rng(seed=iteration)
        angles = rng.uniform(0, 2 * np.pi, size=coincident_mask.sum())
        rows, cols = np.where(coincident_mask)
        ux[rows, cols] = np.cos(angles)
        uy[rows, cols] = np.sin(angles)

    # Split the displacement inversely by weight: w[j] / (w[i] + w[j]).
    # Heavier circles (larger radius) absorb less of the correction.
    pair_w = w[:, None] + w[None, :]  # (n, n)
    pair_w = np.where(pair_w > 0, pair_w, 1.0)

    # Displace circle i away from j (opposite to ux) by overlap * w[j] / pair_w.
    disp_scale = overlap * w[None, :] / pair_w  # (n, n)
    cx = cx - np.sum(ux * disp_scale, axis=1)  # sum contributions over all j
    cy = cy - np.sum(uy * disp_scale, axis=1)

    return cx, cy, max_overlap


def bubble_centers(
    data: pd.DataFrame,
    x: str,
    y: str,
    radius: str,
    buffer: float,
) -> pd.DataFrame:
    """
    Compute non-overlapping circle center positions that approximately minimize
    total weighted displacement from preferred positions.

    Each circle's displacement is weighted by the square of its radius, so
    larger circles are penalized more for moving and will be displaced less.

    The algorithm runs in two phases:

    **Phase 1 – force-directed positioning.**
    Overlapping circles are pushed apart (weighted so heavier circles move
    less), then all circles drift a small step back toward their preferred
    positions.  This phase terminates when the maximum overlap stabilizes,
    which indicates the system has reached a steady state where the separation
    and drift forces are balanced.  When circles must be separated from their
    preferred positions, this balanced state may have a small residual overlap
    (a limit cycle); Phase 2 corrects this.

    **Phase 2 – strict overlap resolution.**
    Pure overlap resolution is applied without any drift.  This guarantees
    that the final positions satisfy all separation constraints exactly,
    independent of how Phase 1 terminated.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing preferred positions and circle sizes.
    x : str
        Column name for the preferred x coordinate of each circle.
    y : str
        Column name for the preferred y coordinate of each circle.
    radius : str
        Column name for the radius of each circle.
    buffer : float
        Minimum gap distance to maintain between any two circles, in addition
        to the sum of their radii.

    Returns
    -------
    pd.DataFrame
        A copy of ``data`` with two additional columns, ``_bubble_x`` and
        ``_bubble_y``, giving the computed non-overlapping circle centers.

    Warns
    -----
    UserWarning
        If Phase 2 cannot satisfy all separation constraints within the
        maximum number of iterations.  This may indicate an over-constrained
        layout (e.g., too many large circles in a confined region).
    """
    n = len(data)

    if n == 0:
        result = data.copy()
        result["_bubble_x"] = pd.Series(dtype=float)
        result["_bubble_y"] = pd.Series(dtype=float)
        return result

    px = data[x].to_numpy(dtype=float)
    py = data[y].to_numpy(dtype=float)
    r = data[radius].to_numpy(dtype=float)

    # Weight = radius²: larger circles resist displacement more.
    w = r**2

    # Minimum required center-to-center distance for every pair of circles.
    min_sep = r[:, None] + r[None, :] + buffer  # shape (n, n)

    # Start with circles at their preferred positions.
    cx = px.copy()
    cy = py.copy()

    tol = 1e-6
    # Fraction to drift toward the preferred position each phase-1 iteration.
    # Must be small enough not to re-introduce large overlaps.
    drift_rate = 0.02

    # ── Phase 1: force-directed (separation + drift) ────────────────────────
    # Run until the maximum overlap stabilizes.  With a non-zero drift rate
    # the system can enter a limit cycle where drift and separation forces
    # exactly balance; detecting this state is the phase-1 stopping criterion.
    prev_max_overlap = np.inf
    for it in range(1000):
        cx, cy, max_overlap = _resolve_overlaps(cx, cy, w, min_sep, iteration=it)

        # Drift each circle a small step toward its preferred position.
        cx = cx + drift_rate * (px - cx)
        cy = cy + drift_rate * (py - cy)

        # Stop when the maximum overlap has stabilized (limit cycle detected
        # or full convergence), meaning phase 1 has done all it can.
        if abs(max_overlap - prev_max_overlap) < tol:
            break
        prev_max_overlap = max_overlap

    # ── Phase 2: strict overlap resolution (no drift) ───────────────────────
    # Eliminate any remaining overlaps without moving circles toward preferred
    # positions.  This guarantees constraint satisfaction in the output.
    converged = False
    max_overlap = np.inf
    for it in range(1000):
        cx, cy, max_overlap = _resolve_overlaps(cx, cy, w, min_sep, iteration=it)
        if max_overlap < tol:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"bubble_centers did not converge after the maximum number of "
            f"iterations.  Maximum remaining overlap: {max_overlap:.4g}. "
            "The result may not satisfy all separation constraints. "
            "The layout may be over-constrained.",
            stacklevel=2,
        )

    result = data.copy()
    result["_bubble_x"] = cx
    result["_bubble_y"] = cy
    return result
