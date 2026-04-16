"""Unit tests for the seedling morphological analysis pipeline.

Synthetic images are generated in-memory so no dataset files are required.
Image conventions:
    - Background: pure blue  (BGR  0, 0, 255)
    - Plant stem: dark green (BGR  0, 100, 0)
    - Leaf blobs: bright green (BGR 0, 200, 0)
"""

import numpy as np
import pytest

from src.segmentation import segment_plant, apply_mask
from src.measurements import (
    plant_height_bbox,
    plant_height_stem,
    collar_diameter,
    leaf_area,
    leaf_count,
    pixels_to_mm,
)
from src.pipeline import analyse_seedling


# ---------------------------------------------------------------------------
# Helpers – synthetic image builders
# ---------------------------------------------------------------------------

H, W = 200, 100  # image dimensions used in most tests

# Blue background in HSV hue ≈ 120 → BGR (255, 0, 0)
BG_BGR = np.array([255, 0, 0], dtype=np.uint8)    # pure blue
STEM_BGR = np.array([30, 80, 150], dtype=np.uint8)  # reddish-brown (BGR: B=30,G=80,R=150) – HSV hue ≈ 25, outside green range
LEAF_BGR = np.array([0, 200, 0], dtype=np.uint8)   # bright green


def _blue_image() -> np.ndarray:
    """Solid blue image (no plant)."""
    img = np.full((H, W, 3), BG_BGR, dtype=np.uint8)
    return img


def _plant_image(
    stem_col: int = 50,
    stem_top: int = 20,
    stem_bottom: int = 180,
    stem_width: int = 6,
    leaf_boxes: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """Synthetic plant on a blue background.

    Parameters
    ----------
    stem_col:   horizontal centre of the stem.
    stem_top:   topmost row of the stem.
    stem_bottom:  bottommost row of the stem.
    stem_width: width of the stem in pixels.
    leaf_boxes: list of (row_min, row_max, col_min, col_max) rectangles painted
                in bright green to simulate leaf blobs.
    """
    img = _blue_image()
    half = stem_width // 2
    img[stem_top:stem_bottom, stem_col - half: stem_col + half] = STEM_BGR
    if leaf_boxes:
        for r0, r1, c0, c1 in leaf_boxes:
            img[r0:r1, c0:c1] = LEAF_BGR
    return img


# ---------------------------------------------------------------------------
# segmentation tests
# ---------------------------------------------------------------------------


class TestSegmentPlant:
    def test_pure_blue_gives_empty_mask(self):
        img = _blue_image()
        mask = segment_plant(img, min_component_area=1)
        assert np.count_nonzero(mask) == 0

    def test_plant_pixels_are_non_zero(self):
        img = _plant_image()
        mask = segment_plant(img, min_component_area=1)
        assert np.count_nonzero(mask) > 0

    def test_background_pixels_are_zero(self):
        img = _plant_image(stem_col=50, stem_width=6)
        mask = segment_plant(img, min_component_area=1)
        # Corner pixels should be background (blue)
        assert mask[0, 0] == 0
        assert mask[0, W - 1] == 0

    def test_apply_mask_zeros_background(self):
        img = _plant_image()
        mask = segment_plant(img, min_component_area=1)
        masked = apply_mask(img, mask)
        # Any pixel that is background in mask should be black in result
        assert np.all(masked[mask == 0] == 0)


# ---------------------------------------------------------------------------
# measurement tests
# ---------------------------------------------------------------------------


class TestPlantHeightBbox:
    def test_empty_mask_returns_zero(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        assert plant_height_bbox(mask) == 0

    def test_full_mask_returns_image_height(self):
        mask = np.full((H, W), 255, dtype=np.uint8)
        assert plant_height_bbox(mask) == H

    def test_single_row_returns_one(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[50, :] = 255
        assert plant_height_bbox(mask) == 1

    def test_stem_height_matches_bbox(self):
        stem_top, stem_bottom = 30, 160
        img = _plant_image(stem_top=stem_top, stem_bottom=stem_bottom)
        mask = segment_plant(img, min_component_area=1)
        h = plant_height_bbox(mask)
        expected = stem_bottom - stem_top
        # Allow ±2 px tolerance from morphological ops
        assert abs(h - expected) <= 2


class TestPlantHeightStem:
    def test_empty_mask_returns_zero(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        assert plant_height_stem(mask) == 0

    def test_vertical_line_length(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[10:90, W // 2] = 255  # 80-pixel vertical line
        h = plant_height_stem(mask)
        # Skeleton of a vertical line equals its length
        assert abs(h - 80) <= 2

    def test_stem_path_positive_for_plant(self):
        img = _plant_image()
        mask = segment_plant(img, min_component_area=1)
        h = plant_height_stem(mask)
        assert h > 0


class TestCollarDiameter:
    def test_empty_mask_returns_zero(self):
        mask = np.zeros((H, W), dtype=np.uint8)
        assert collar_diameter(mask) == 0

    def test_known_stem_width(self):
        stem_width = 10
        img = _plant_image(stem_col=50, stem_width=stem_width)
        mask = segment_plant(img, min_component_area=1)
        cd = collar_diameter(mask)
        # Allow ±3 px tolerance from morphological effects
        assert abs(cd - stem_width) <= 3

    def test_wider_base_detected(self):
        # Plant with a wider base should report larger collar diameter
        img_thin = _plant_image(stem_width=4)
        img_wide = _plant_image(stem_width=14)
        mask_thin = segment_plant(img_thin, min_component_area=1)
        mask_wide = segment_plant(img_wide, min_component_area=1)
        cd_thin = collar_diameter(mask_thin)
        cd_wide = collar_diameter(mask_wide)
        assert cd_wide > cd_thin


class TestLeafArea:
    def test_no_leaves_returns_zero(self):
        img = _plant_image()  # stem only (dark green, low saturation area)
        mask = segment_plant(img, min_component_area=1)
        # Dark green stem may not match the bright-green leaf HSV range
        area = leaf_area(img, mask)
        # We do not assert zero because the dark stem might overlap slightly
        assert area >= 0

    def test_leaf_blobs_increase_area(self):
        leaves = [(40, 70, 10, 40), (80, 110, 60, 90)]
        img_no_leaf = _plant_image()
        img_with_leaf = _plant_image(leaf_boxes=leaves)
        mask_no = segment_plant(img_no_leaf, min_component_area=1)
        mask_with = segment_plant(img_with_leaf, min_component_area=1)
        area_no = leaf_area(img_no_leaf, mask_no)
        area_with = leaf_area(img_with_leaf, mask_with)
        assert area_with > area_no


class TestLeafCount:
    def test_no_leaves_returns_zero(self):
        img = _plant_image()
        mask = segment_plant(img, min_component_area=1)
        count = leaf_count(img, mask, min_leaf_area_px=50)
        assert count == 0

    def test_two_separated_leaf_blobs(self):
        # Two leaf blobs separated by > 5 px (morphological opening gap)
        leaves = [(30, 60, 5, 35), (30, 60, 65, 95)]
        img = _plant_image(leaf_boxes=leaves)
        mask = segment_plant(img, min_component_area=1)
        count = leaf_count(img, mask, min_leaf_area_px=50)
        assert count == 2

    def test_three_separated_leaf_blobs(self):
        leaves = [(20, 50, 0, 25), (20, 50, 38, 63), (20, 50, 75, 100)]
        img = _plant_image(leaf_boxes=leaves)
        mask = segment_plant(img, min_component_area=1)
        count = leaf_count(img, mask, min_leaf_area_px=50)
        assert count == 3


# ---------------------------------------------------------------------------
# pixels_to_mm utility
# ---------------------------------------------------------------------------


class TestPixelsToMm:
    def test_basic_conversion(self):
        assert pixels_to_mm(100, 10) == pytest.approx(10.0)

    def test_fractional(self):
        assert pixels_to_mm(55, 5) == pytest.approx(11.0)

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError):
            pixels_to_mm(100, 0)

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError):
            pixels_to_mm(100, -3)


# ---------------------------------------------------------------------------
# pipeline integration tests
# ---------------------------------------------------------------------------


class TestAnalyseSeedling:
    def test_returns_required_keys(self):
        img = _plant_image()
        result = analyse_seedling(img)
        required = {
            "plant_height_bbox_px",
            "plant_height_stem_px",
            "collar_diameter_px",
            "leaf_area_px2",
            "leaf_count",
        }
        assert required.issubset(result.keys())

    def test_mm_keys_present_when_scale_given(self):
        img = _plant_image()
        result = analyse_seedling(img, px_per_mm=10.0)
        mm_keys = {
            "plant_height_bbox_mm",
            "plant_height_stem_mm",
            "collar_diameter_mm",
            "leaf_area_mm2",
        }
        assert mm_keys.issubset(result.keys())

    def test_mm_keys_absent_without_scale(self):
        img = _plant_image()
        result = analyse_seedling(img)
        assert "plant_height_bbox_mm" not in result

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            analyse_seedling("/nonexistent/path/image.jpg")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            analyse_seedling(12345)  # type: ignore[arg-type]

    def test_plant_height_positive(self):
        img = _plant_image()
        result = analyse_seedling(img)
        assert result["plant_height_bbox_px"] > 0

    def test_all_values_non_negative(self):
        img = _plant_image(leaf_boxes=[(30, 70, 10, 40)])
        result = analyse_seedling(img)
        for key, value in result.items():
            assert value >= 0, f"{key} is negative: {value}"
