import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from library.selectBlob import *
from seedling_inspector import masks, remove_bg

paths_eucalyptus = [f"_Eucalipto_Escolhidos1/Eucalipto{i}.jpg"
         for i in range(1, 6)]
imgs_eucalyptus = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths_eucalyptus]

paths_pines = [f"_Pinheiro_Escolhidos1/Pinheiro{i}.jpg"
         for i in range(1, 4)]
imgs_pines = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths_pines]



def leaf_area(img):
    plant = masks(img)[0]

    kw = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))

    img_no_bg = remove_bg(img)

    img_segmentada = cv2.morphologyEx(plant, cv2.MORPH_OPEN,  kw)

    area = np.sum(img_segmentada == 255)
    
    return area


print('Área das folhas de eucalipto')
for i in range(len(imgs_eucalyptus)):
    print('Imagem', i + 1, ':', print(leaf_area(imgs_eucalyptus[i])))