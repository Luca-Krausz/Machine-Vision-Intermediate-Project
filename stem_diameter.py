<<<<<<< HEAD
=======
# Basico
>>>>>>> e6a1cd451a4c88acfb39c1552d085f45270f364d
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from library.selectBlob import *
<<<<<<< HEAD
from seedling_inspector import masks

paths = [f"_Eucalipto_Escolhidos1/Eucalipto{i}.jpg" for i in range(1, 6)]
imgs_eu = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
plants_eu = [masks(img)[1] for img in imgs_eu]

plt.imshow(plants_eu[0], cmap='gray')
plt.show()

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
=======
from seedling_inspector import remove_bg, masks

paths_eucalyptus = [f"_Eucalipto_Escolhidos1/Eucalipto{i}.jpg"
         for i in range(1, 6)]
imgs_eucalyptus = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths_eucalyptus]

paths_pines = [f"_Pinheiro_Escolhidos1/Pinheiro{i}.jpg"
         for i in range(1, 4)]
imgs_pines = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths_pines]


plant_eucalyptus = masks(imgs_eucalyptus[0])[1]

ys, xs = np.where(plant_eucalyptus == 255)

y_max = np.max(ys)

x_base = xs[ys == y_max]
print(x_base)

diameter = np.max(x_base) - np.min(x_base)

plt.figure()
plt.imshow(plant_eucalyptus, cmap ='gray')
plt.show()



# diameter = np.max(x_base) - np.min(x_base)

# print(diameter)

# length = y_max - y_min



# print('Comprimento do caule da muda de eucalipto')
# for i in range(len(imgs_eucalyptus)):
#     print('Imagem', i + 1, ':',
#           stem_length(imgs_eucalyptus[i]), 'pixels')

# print('Comprimento do caule da muda de pinheiro')
# for j in range(len(imgs_pines)):
#     print('Imagem', j + 1, ':',
#           stem_length(imgs_pines[j]), 'pixels')
>>>>>>> e6a1cd451a4c88acfb39c1552d085f45270f364d
