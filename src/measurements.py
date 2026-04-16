"""Morphological measurements extracted from a segmented seedling image.

All measurements that return a pixel count or pixel distance are accompanied by
a ``_px`` suffix in the returned dictionary so callers can apply their own
pixels-per-millimetre scale factor.  Helper ``pixels_to_mm`` is provided for
convenience when the physical scale is known.

Measurements implemented
------------------------
- **plant_height_bbox**  – height of the plant bounding box (pixels).
- **plant_height_stem**  – length of the skeletonised stem path (pixels).
- **collar_diameter**    – width of the stem at the collar region (pixels).
- **leaf_area**          – total green-pixel area (pixels²).
- **leaf_count**         – number of individual leaf blobs detected.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize


# Default HSV range used to isolate green leaf tissue.
_GREEN_LOW = np.array([35, 40, 40], dtype=np.uint8)
_GREEN_HIGH = np.array([90, 255, 255], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plant_height_bbox(plant_mask: np.ndarray) -> int:
    """Return the height of the axis-aligned bounding box of *plant_mask* (px).

    Parameters
    ----------
    plant_mask:
        Binary mask (0/255) where plant pixels are 255.

    Returns
    -------
    int
        Height in pixels, or 0 if the mask contains no plant pixels.
    """
    ys, _ = np.where(plant_mask > 0)
    if len(ys) == 0:
        return 0
    return int(ys.max() - ys.min() + 1)


def plant_height_stem(plant_mask: np.ndarray) -> int:
    """Estimate plant height by tracing the skeletonised stem path (px).

    The function skeletonises the plant mask and counts the number of pixels
    in the longest connected path of the skeleton, which approximates the
    arc-length of the main stem from root to tip.

    Parameters
    ----------
    plant_mask:
        Binary mask (0/255) where plant pixels are 255.

    Returns
    -------
    int
        Path length in pixels, or 0 if the mask is empty.
    """
    if not np.any(plant_mask > 0):
        return 0

    binary = (plant_mask > 0).astype(np.uint8)
    skeleton = skeletonize(binary).astype(np.uint8)

    # Count pixels in the longest connected component of the skeleton
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton, connectivity=8
    )
    if n_labels <= 1:
        return 0

    longest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return longest


def collar_diameter(
    plant_mask: np.ndarray,
    collar_fraction: float = 0.15,
) -> int:
    """Estimate collar diameter as the stem width near the base of the plant.

    The collar is assumed to be located at *collar_fraction* of the plant
    height measured from the bottom of the bounding box.  The width is
    computed as the horizontal extent of plant pixels at that row.

    Parameters
    ----------
    plant_mask:
        Binary mask (0/255) where plant pixels are 255.
    collar_fraction:
        Fraction of plant height from the bottom at which to measure width.
        Default is 0.15 (15 % from the bottom).

    Returns
    -------
    int
        Collar width in pixels, or 0 if the mask is empty.
    """
    ys, xs = np.where(plant_mask > 0)
    if len(ys) == 0:
        return 0

    y_max = int(ys.max())
    y_min = int(ys.min())
    height = y_max - y_min
    if height == 0:
        return int(xs.max() - xs.min() + 1)

    # Row index near the base
    target_row = y_max - max(1, int(height * collar_fraction))

    # Find the plant pixels in a narrow band around the target row
    band_half = max(2, int(height * 0.02))
    band_mask = plant_mask[
        max(0, target_row - band_half): target_row + band_half + 1, :
    ]
    col_pixels = np.where(band_mask > 0)[1]
    if len(col_pixels) == 0:
        return 0
    return int(col_pixels.max() - col_pixels.min() + 1)


def leaf_area(
    image_bgr: np.ndarray,
    plant_mask: np.ndarray,
    green_low: np.ndarray = _GREEN_LOW,
    green_high: np.ndarray = _GREEN_HIGH,
) -> int:
    """Return total leaf area as the number of green pixels inside *plant_mask*.

    Parameters
    ----------
    image_bgr:
        Source image in BGR colour order.
    plant_mask:
        Binary mask (0/255) restricting the search to plant pixels.
    green_low:
        Lower HSV bound for green leaf tissue.
    green_high:
        Upper HSV bound for green leaf tissue.

    Returns
    -------
    int
        Number of green pixels (proxy for leaf area in px²).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, green_low, green_high)
    combined = cv2.bitwise_and(green_mask, plant_mask)
    return int(np.count_nonzero(combined))


def leaf_count(
    image_bgr: np.ndarray,
    plant_mask: np.ndarray,
    green_low: np.ndarray = _GREEN_LOW,
    green_high: np.ndarray = _GREEN_HIGH,
    min_leaf_area_px: int = 200,
) -> int:
    """Count individual leaves using connected-component analysis.

    Parameters
    ----------
    image_bgr:
        Source image in BGR colour order.
    plant_mask:
        Binary mask (0/255) restricting the search to plant pixels.
    green_low:
        Lower HSV bound for green leaf tissue.
    green_high:
        Upper HSV bound for green leaf tissue.
    min_leaf_area_px:
        Minimum component size (pixels) to be counted as a leaf.

    Returns
    -------
    int
        Number of detected leaves.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, green_low, green_high)
    combined = cv2.bitwise_and(green_mask, plant_mask)

    # Light morphological opening to separate touching leaves
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )
    count = 0
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_leaf_area_px:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def pixels_to_mm(pixels: float, px_per_mm: float) -> float:
    """Convert a pixel measurement to millimetres.

    Parameters
    ----------
    pixels:
        Measurement in pixels.
    px_per_mm:
        Spatial scale: number of pixels per millimetre.

    Returns
    -------
    float
        Equivalent measurement in millimetres.
    """
    if px_per_mm <= 0:
        raise ValueError(f"px_per_mm must be positive, got {px_per_mm}")
    return pixels / px_per_mm
