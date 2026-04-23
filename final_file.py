# Comprimento básico
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from library.selectBlob import *
from main import remove_bg, masks
from skimage.morphology import skeletonize

# Comprimento do caule
paths_eucalyptus = [f"_Eucalipto_Escolhidos1/Eucalipto{i}.jpg"
         for i in range(1, 6)]
imgs_eucalyptus = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths_eucalyptus]

paths_pines = [f"_Pinheiro_Escolhidos1/Pinheiro{i}.jpg"
         for i in range(1, 4)]
imgs_pines = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths_pines]

def stem_length(img):
    plant_eucalyptus = masks(img)[1]

    ys, xs = np.where(plant_eucalyptus == 255)

    y_min = np.min(ys)
    y_max = np.max(ys)

    length = y_max - y_min

    return length


# Diametro do colo
paths = [f"_Eucalipto_Escolhidos1/Eucalipto{i}.jpg" for i in range(1, 6)]
imgs_eu = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
plants_eu = [masks(img)[1] for img in imgs_eu]


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



# Área foliar eucalipto

def leaf_area_eucalyptus(img):
    plant = masks(img)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))

    img_segmentada = cv2.morphologyEx(plant, cv2.MORPH_OPEN,  kernel)

    area = np.sum(img_segmentada == 255)
    
    return area

    



  
# Área foliar pinheiros

def leaf_area_pines(img):
    # pot = masks(img)[0]
    
    stem_skel = (skeletonize(masks(img)[1] > 0) * 255).astype(np.uint8)
    
    _, lbl, stats, _ = cv2.connectedComponentsWithStats(stem_skel, 8)
    img_filter1 = np.where(lbl == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
    skel_dilated = cv2.dilate(img_filter1, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    trunk = cv2.bitwise_and(masks(img)[1], cv2.bitwise_not(skel_dilated))
    
    img_sat = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 2]
    img_sat_filter = np.where((img_sat > 100) & (img_sat < 150), 255, 0).astype(np.uint8)
    
    img_cutted = img_sat_filter.copy()
    img_cutted[2450:, :] = 0
    
    img_wo_trunk = cv2.bitwise_and(img_cutted, cv2.bitwise_not(trunk))
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    img_vert = cv2.morphologyEx(img_wo_trunk, cv2.MORPH_OPEN, vertical_kernel)
    
    img_wo_vert = cv2.bitwise_and(img_wo_trunk, cv2.bitwise_not(img_vert))
    
    leaf_area = np.sum(img_wo_vert == 255)
    
    return leaf_area, img_wo_vert



dados_eu = []

for i in range(len(imgs_eucalyptus)):
    img = imgs_eucalyptus[i]
    plant = masks(img)[1]

    altura = stem_length(img)
    diametro, _ = collar_diameter(plant)
    area = leaf_area_eucalyptus(img)

    dados_eu.append([i + 1, altura, diametro, area])

df_eu = pd.DataFrame(dados_eu, columns=['Img', 'Altura', 'Diâmetro', 'Área'])

df_eu.to_csv('eucalipto.csv', index=False, sep=';')

print('\nCSV Eucalipto:')
print(df_eu)


dados_pine = []

for i in range(len(imgs_pines)):
    img = imgs_pines[i]
    plant = masks(img)[1]

    altura = stem_length(img)
    diametro, _ = collar_diameter(plant)
    area = leaf_area_pines(img)[0]

    dados_pine.append([i + 1, altura, diametro, area])

df_pine = pd.DataFrame(dados_pine, columns=['Img', 'Altura', 'Diâmetro', 'Área'])

df_pine.to_csv('pinheiro.csv', index=False, sep=';')

print('\nCSV Pinheiro:')
print(df_pine)