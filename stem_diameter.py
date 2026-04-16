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