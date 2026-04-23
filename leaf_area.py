import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from library.selectBlob import *
from skimage.filters import meijering
from main import masks, remove_bg, skeletonize

paths_eucalyptus = [f"_Eucalipto_Escolhidos1/Eucalipto{i}.jpg"
         for i in range(1, 6)]
imgs_eucalyptus = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths_eucalyptus]

paths_pines = [f"_Pinheiro_Escolhidos1/Pinheiro{i}.jpg"
         for i in range(1, 4)]
imgs_pines = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths_pines]



def leaf_area_eucalyptus(img):
    plant = masks(img)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))

    img_segmentada = cv2.morphologyEx(plant, cv2.MORPH_OPEN,  kernel)

    area = np.sum(img_segmentada == 255)
    
    return area


print('Área das folhas de eucalipto')

for i in range(len(imgs_eucalyptus)):
    area = leaf_area_eucalyptus(imgs_eucalyptus[i])
    print('Imagem', i + 1, ':', area)

    
    
# Leaf area for pines

def leaf_area_pines(img):
    # pot = masks(img)[0]
    
    stem_skel = skeletonize(masks(img)[1])
    
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

for i in range(len(imgs_pines)):
    print(f'Área das folhas de pinheiro da imagem {i + 1}: {leaf_area_pines(imgs_pines[i])[0]}')

plt.figure(figsize=(15, 5))
for i in range(len(imgs_pines)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(leaf_area_pines(imgs_pines[i])[1], cmap='gray')
    plt.title(f'Pinheiro {i + 1}')
plt.tight_layout()
plt.show()