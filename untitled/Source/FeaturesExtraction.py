import cv2
import numpy as np
import matplotlib.pyplot as plt

# Rt = 88
# Gt = 61
# Bt = 86
# Gt = 70

def Append(vector, number):
    # number = np.round(number, 5)
    vector.append(number)
    return

def GetFeatureVector(feature_vector, image):
    [w, h] = image.shape

    nz = np.count_nonzero(image)
    Append(feature_vector, nz)

    x = np.sum(image, axis=0)
    Append(feature_vector, np.mean(x))
    Append(feature_vector, np.std(x))
    y = np.sum(image, axis=1)
    Append(feature_vector, np.mean(y))
    Append(feature_vector, np.std(y))

    Append(feature_vector, np.sum(image))
    return

def Threshold(image, t):
    tmp = image.copy()
    thresh = tmp >= t
    # tmp[thresh] = 1
    tmp[~thresh] = 0
    tmp[:12,:] = 0
    return tmp

def Extraction(image):
    image = cv2.resize(image, (231, 45))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    THRESH = [88, 61, 86, 70]
    feature_vector = []

    gray = Threshold(gray, THRESH[3])
    GetFeatureVector(feature_vector, gray)
    for i in range(image.shape[2]):
        tmp_image = Threshold(image[:, :, i], THRESH[i])
        GetFeatureVector(feature_vector, tmp_image)

    feature_vector = np.array(feature_vector)
    # feature_vector = np.interp(feature_vector, (feature_vector.min(), feature_vector.max()), (0, 1))

    return feature_vector