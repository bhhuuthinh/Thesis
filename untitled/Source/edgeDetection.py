import pandas as pd
from os import listdir
from os.path import join, isfile
import cv2
from numpy import *
import numpy as np
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from scipy import ndimage

def IM_PROCESSING(image):
    img_blur = cv2.blur(image,(3,3))
    # sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)  # x
    # sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)  # y

    sobelxM = np.array( [[ 0, 0, 0 ],
                             [ 0, 1, 0 ],
                             [ 0, 0,-1 ]] )

    sobelyM = np.array( [[ 0, 0, 0 ],
                             [ 0, 0, 1 ],
                             [ 0,-1, 0 ]] )


    vertical = ndimage.convolve(img_blur, sobelxM)
    horizontal = ndimage.convolve(img_blur, sobelyM)

    sobel = np.sqrt(np.square(horizontal) + np.square(vertical))

    # sobel = sqrt(sobelx ** 2 + sobely ** 2)

    # MAX = max(sobel.ravel())
    # thresh = sobel <= MAX / 5
    # sobel[thresh] = 0
    # sobel[~thresh] = 1

    return np.uint8(sobel)

PATH = "C:\\Users\\thanh\\Downloads"
img = cv2.imread(PATH + "/76375976.jpg", 0)
img = IM_PROCESSING(img)

img = 255 - img
thresh = img >= 250
img[thresh] = 255
img[~thresh] = 0

plt.imshow(img, cmap='gray')
plt.show()
