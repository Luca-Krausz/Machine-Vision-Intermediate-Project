import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from library.selectBlob import *



img_eu1_base = cv2.cvtColor(cv2.imread("Machine-Vision-Intermediate-Project/_Eucalipto_Escolhidos1/Eucalipto1.jpg"), cv2.COLOR_BGR2RGB)

if img_eu1_base is None:
    print("Error: Image not found.")

def remove_bg(img):
    img_no_bg = img.copy()
    img_no_bg[img_no_bg[:, :, 2] > 140] = (0, 0, 0)
    return img_no_bg

def collar_diameter(img):
    img_no_bg = remove_bg(img)
    img_v = cv2.cvtColor(img_no_bg, cv2.COLOR_RGB2HSV_FULL)[..., 2]
    
    # Binarization # 
    
    bin_treashhold = cv2.inRange(img_v, 0, 255)
    
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    
    params.filterByArea = True
    params.minArea = 10000
    params.maxArea = 1000000
    
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    
    detector = cv2.SimpleBlobDetector_create(params)
    

img_eu1_p = cv2.cvtColor(img_eu1_base, cv2.COLOR_RGB2HSV_FULL)[..., 2]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(remove_bg(img_eu1_base), cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(img_eu1_p, cmap='gray')
plt.show()