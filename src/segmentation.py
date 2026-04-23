"""Blue-background segmentation for seedling images.

Images are captured against a controlled blue backdrop.  The plant mask is
obtained by thresholding the Hue channel in HSV space and inverting the result,
followed by morphological clean-up to remove small artefacts.
"""

import cv2
import numpy as np


# Default HSV range for the blue background.
# Hue 90–130 covers most shades of blue/cyan under typical studio lighting.
_BLUE_HUE_LOW = np.array([90, 50, 50], dtype=np.uint8)
_BLUE_HUE_HIGH = np.array([130, 255, 255], dtype=np.uint8)


def segment_plant(
    image_bgr: np.ndarray,
    blue_low: np.ndarray = _BLUE_HUE_LOW,
    blue_high: np.ndarray = _BLUE_HUE_HIGH,
    morph_kernel_size: int = 5,
    min_component_area: int = 500,
) -> np.ndarray:
    """Return a binary mask (uint8, 0/255) isolating the plant from the background.

    Parameters
    ----------
    image_bgr:
        Source image in BGR colour order (as returned by ``cv2.imread``).
    blue_low:
        Lower HSV bound for the blue background (default covers H 90–130).
    blue_high:
        Upper HSV bound for the blue background.
    morph_kernel_size:
        Side length of the square structuring element used for morphological
        opening and closing.
    min_component_area:
        Connected components with fewer pixels than this value are removed as
        noise.

    Returns
    -------
    np.ndarray
        Binary mask of the same spatial dimensions as *image_bgr*, where 255
        indicates plant pixels and 0 indicates background.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Mask of background (blue) pixels
    bg_mask = cv2.inRange(hsv, blue_low, blue_high)

    # Plant mask = inverse of background
    plant_mask = cv2.bitwise_not(bg_mask)

    # Morphological opening removes isolated noise pixels
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)

    # Morphological closing fills small holes inside the plant region
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)

    # Remove small connected components (noise)
    plant_mask = _remove_small_components(plant_mask, min_component_area)

    return plant_mask


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than *min_area* pixels."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    for label in range(1, n_labels):  # skip background label 0
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def apply_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return *image_bgr* with background pixels set to black."""
    masked = image_bgr.copy()
    masked[mask == 0] = 0
    return masked
