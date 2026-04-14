
#------------------------------------------------------------------------------
# Load Dependencies
#------------------------------------------------------------------------------
import cv2
import numpy as np

#------------------------------------------------------------------------------
# Function selectBlob()
#------------------------------------------------------------------------------
def fillHoles(img_thr):

    """
    Prenche furos internos em blobs, que deve ser branco (intensidade=255)

    Keyword arguments:
    img_thr -- nparray uint8 da imagem de entrada (imagem binarizada)

    Returns:
    img_out -- npaarray (uint8) da imagem apos preenchimento dos furos

    """

    img_floodfill = img_thr.copy()
    h, w = img_thr.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(img_floodfill, mask, (0,0), 255)
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)
    img_out = img_thr | img_floodfill_inv

    return img_out
#------------------------------------------------------------------------------