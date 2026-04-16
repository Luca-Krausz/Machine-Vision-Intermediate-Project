# Basico
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from library.selectBlob import *
from seedling_inspector import remove_bg, masks

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

print('Comprimento do caule da muda de eucalipto')
for i in range(len(imgs_eucalyptus)):
    print('Imagem', i + 1, ':',
          stem_length(imgs_eucalyptus[i]), 'pixels')

print('Comprimento do caule da muda de pinheiro')
for j in range(len(imgs_pines)):
    print('Imagem', j + 1, ':',
          stem_length(imgs_pines[j]), 'pixels')