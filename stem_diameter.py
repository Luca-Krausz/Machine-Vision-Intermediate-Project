import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from library.selectBlob import *
from main import masks

# Eucalyptus
paths = [f"_Eucalipto_Escolhidos1/Eucalipto{i}.jpg" for i in range(1, 6)]
imgs_eu = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
plants_eu = [masks(img)[1] for img in imgs_eu]

# Pines
paths = [f"_Pinheiro_Escolhidos1/Pinheiro{i}.jpg" for i in range(1, 4)]
imgs_pines = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
plants_pines = [masks(img)[1] for img in imgs_pines]

def collar_diameter(plant_mask):
    white_rows = np.where(np.any(plant_mask == 255, axis=1))[0]
    if len(white_rows) == 0:
        return 0, (0, 0, 0)

    bottom_white_row = int(white_rows[-1])
    cut_end = bottom_white_row - 10
    if cut_end <= 0:
        return 0, (0, 0, 0)
    cropped = plant_mask[:cut_end, :]

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (cropped > 0).astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return 0, (0, 0, 0)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    plant = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    w = plant.shape[1]
    strip_width = 200 if w % 2 == 0 else 201
    x_start = (w - strip_width) // 2
    x_end = x_start + strip_width
    strip = plant[:, x_start:x_end]

    strip_rows = np.where(np.any(strip == 255, axis=1))[0]
    if len(strip_rows) == 0:
        return 0, (0, 0, 0)
    y_collar = int(strip_rows[-1])

    xs = np.where(strip[y_collar] == 255)[0]
    x_min = int(xs.min()) + x_start
    x_max = int(xs.max()) + x_start

    return x_max - x_min, (y_collar, x_min, x_max)

# Seeing the results for eucalyptus

plt.figure(figsize=(5 * len(plants_eu), 10))
for i, (img, plant) in enumerate(zip(imgs_eu, plants_eu), start=1):
    diam, (y, x1, x2) = collar_diameter(plant)

    vis = cv2.cvtColor(plant, cv2.COLOR_GRAY2RGB)
    cv2.line(vis, (x1, y), (x2, y), (0, 255, 0), 3)

    
    plt.subplot(2, len(plants_eu), i)
    plt.imshow(vis)
    plt.title(f"Euc {i} - collar = {diam} px")
    plt.axis("off")

    plt.subplot(2, len(plants_eu), len(plants_eu) + i)
    plt.imshow(img)
    plt.title(f"original {i}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# Seeing the results for pines

plt.figure(figsize=(5 * len(plants_pines), 10))
for i, (img, plant) in enumerate(zip(imgs_pines, plants_pines), start=1):
    diam, (y, x1, x2) = collar_diameter(plant)

    vis = cv2.cvtColor(plant, cv2.COLOR_GRAY2RGB)
    cv2.line(vis, (x1, y), (x2, y), (0, 255, 0), 3)

    
    plt.subplot(2, len(plants_pines), i)
    plt.imshow(vis)
    plt.title(f"Pine {i} - collar = {diam} px")
    plt.axis("off")

    plt.subplot(2, len(plants_pines), len(plants_pines) + i)
    plt.imshow(img)
    plt.title(f"original {i}")
    plt.axis("off")
    
plt.tight_layout()
plt.show()