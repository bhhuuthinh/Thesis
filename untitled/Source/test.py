import Directory
from Dataset import Dataset
from Video import Video
import FeaturesExtraction as FE
from Warehouse import Warehouse
import matplotlib.pyplot as plt

import cv2
import numpy as np
from joblib import dump, load
import operator

if __name__ == "__main__":
	img1 = cv2.imread('E:/Clustered/TT+P. Nedved/0000115784.jpg', 1)
	f1 = FE.Extraction(img1)
	img2 = cv2.imread('C:/Users/thanh/Pictures/Picture2.png', 1)
	f2 = FE.Extraction(img2)
	print(f1)
	print(f2)
	print(np.linalg.norm(f1-f2))