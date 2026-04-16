import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from library.selectBlob import *

paths = [f"Machine-Vision-Intermediate-Project/_Eucalipto_Escolhidos1/Eucalipto{i}.jpg"
         for i in range(1, 6)]
imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]

for i, img in enumerate(imgs):
    if img is None:
        print(f"Error: Image {i+1} not found.")

def remove_bg(img):
    img_no_bg = img.copy()
    img_no_bg[img_no_bg[:, :, 2] > 140] = (0, 0, 0)
    return img_no_bg

def masks(img):
    k  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kw = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 5))

    # Background → foreground silhouette
    img_no_bg = remove_bg(img)
    gray = cv2.cvtColor(img_no_bg, cv2.COLOR_RGB2GRAY)
    _, fg_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)   # foreground=255

    body = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kw)
    body = cv2.morphologyEx(body,    cv2.MORPH_CLOSE, k)

    # Pot = lowest surviving foreground blob
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(body, 8)
    if n <= 1:
        return fg_mask, np.zeros_like(fg_mask)
    bottoms = stats[1:, cv2.CC_STAT_TOP] + stats[1:, cv2.CC_STAT_HEIGHT]
    idx = 1 + int(np.argmax(bottoms))
    pot = np.where(lbl == idx, 255, 0).astype(np.uint8)

    # Plant = foreground above the pot's top row
    rows = np.where(pot.any(axis=1))[0]
    soil_row = int(rows[0]) if len(rows) else fg_mask.shape[0]
    plant = fg_mask.copy()
    plant[soil_row:, :] = 0

    return plant, pot


plant, pot = masks(imgs[0])

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(pot,   cmap='gray'); plt.title('pot')
plt.subplot(1, 3, 2); plt.imshow(plant, cmap='gray'); plt.title('plant')
plt.subplot(1, 3, 3); plt.imshow(remove_bg(imgs[0]), cmap='gray'); plt.title('plant')
plt.show()