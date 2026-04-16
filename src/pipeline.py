"""Main pipeline: orchestrate segmentation and measurement for a single image.

Typical usage
-------------
>>> from src.pipeline import analyse_seedling
>>> result = analyse_seedling("path/to/image.jpg")
>>> print(result)
{
    'plant_height_bbox_px': 542,
    'plant_height_stem_px': 498,
    'collar_diameter_px': 18,
    'leaf_area_px2': 31204,
    'leaf_count': 7,
}

When a known spatial scale is available (pixels per millimetre) supply
*px_per_mm* to obtain additional ``_mm`` / ``_mm2`` keys.
"""

from __future__ import annotations

import cv2
import numpy as np

from .segmentation import segment_plant
from .measurements import (
    plant_height_bbox,
    plant_height_stem,
    collar_diameter,
    leaf_area,
    leaf_count,
    pixels_to_mm,
)


def analyse_seedling(
    image_source: str | np.ndarray,
    px_per_mm: float | None = None,
    blue_low: np.ndarray | None = None,
    blue_high: np.ndarray | None = None,
    morph_kernel_size: int = 5,
    min_component_area: int = 500,
    collar_fraction: float = 0.15,
    min_leaf_area_px: int = 200,
) -> dict:
    """Run the full morphological analysis pipeline on a seedling image.

    Parameters
    ----------
    image_source:
        Either a filesystem path to an image file, or an already-loaded BGR
        NumPy array (as returned by ``cv2.imread``).
    px_per_mm:
        Optional spatial scale in pixels per millimetre.  When provided, the
        returned dictionary includes ``*_mm`` keys with physical measurements.
    blue_low:
        Lower HSV bound for the blue background (passed to
        :func:`~src.segmentation.segment_plant`).
    blue_high:
        Upper HSV bound for the blue background.
    morph_kernel_size:
        Morphological kernel size for segmentation clean-up.
    min_component_area:
        Minimum connected-component area kept after segmentation.
    collar_fraction:
        Fraction of plant height from the base used to measure collar width.
    min_leaf_area_px:
        Minimum component size (px) counted as an individual leaf.

    Returns
    -------
    dict
        Measurement dictionary with at least the following keys:

        - ``plant_height_bbox_px`` – bounding-box plant height (px)
        - ``plant_height_stem_px`` – skeleton stem-path length (px)
        - ``collar_diameter_px``   – collar width (px)
        - ``leaf_area_px2``        – total green-pixel area (px²)
        - ``leaf_count``           – number of detected leaves

        If *px_per_mm* is supplied the dictionary additionally contains:

        - ``plant_height_bbox_mm``
        - ``plant_height_stem_mm``
        - ``collar_diameter_mm``
        - ``leaf_area_mm2``
    """
    # ------------------------------------------------------------------
    # 1. Load image
    # ------------------------------------------------------------------
    if isinstance(image_source, str):
        image_bgr = cv2.imread(image_source)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not load image: {image_source}")
    elif isinstance(image_source, np.ndarray):
        image_bgr = image_source
    else:
        raise TypeError(
            f"image_source must be a path string or NumPy array, "
            f"got {type(image_source)}"
        )

    # ------------------------------------------------------------------
    # 2. Segment plant from blue background
    # ------------------------------------------------------------------
    seg_kwargs: dict = {
        "morph_kernel_size": morph_kernel_size,
        "min_component_area": min_component_area,
    }
    if blue_low is not None:
        seg_kwargs["blue_low"] = blue_low
    if blue_high is not None:
        seg_kwargs["blue_high"] = blue_high

    plant_mask = segment_plant(image_bgr, **seg_kwargs)

    # ------------------------------------------------------------------
    # 3. Compute measurements
    # ------------------------------------------------------------------
    h_bbox = plant_height_bbox(plant_mask)
    h_stem = plant_height_stem(plant_mask)
    collar = collar_diameter(plant_mask, collar_fraction=collar_fraction)
    area = leaf_area(image_bgr, plant_mask)
    count = leaf_count(
        image_bgr, plant_mask, min_leaf_area_px=min_leaf_area_px
    )

    result: dict = {
        "plant_height_bbox_px": h_bbox,
        "plant_height_stem_px": h_stem,
        "collar_diameter_px": collar,
        "leaf_area_px2": area,
        "leaf_count": count,
    }

    # ------------------------------------------------------------------
    # 4. Optionally convert to physical units
    # ------------------------------------------------------------------
    if px_per_mm is not None:
        result["plant_height_bbox_mm"] = round(
            pixels_to_mm(h_bbox, px_per_mm), 2
        )
        result["plant_height_stem_mm"] = round(
            pixels_to_mm(h_stem, px_per_mm), 2
        )
        result["collar_diameter_mm"] = round(
            pixels_to_mm(collar, px_per_mm), 2
        )
        result["leaf_area_mm2"] = round(
            pixels_to_mm(area, px_per_mm ** 2), 2
        )

    return result
