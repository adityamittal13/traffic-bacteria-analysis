# Miscellaneous imports for analysis and data processing
import cv2
import time
import math
import os
from PIL import Image
import numpy as np
from statistics import mean, stdev
import matplotlib.pyplot as plt

# Skimage imports for calculating features
from skimage import measure, color, io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_otsu, gaussian
from skimage import io, color, feature, measure

# Sklearn imports for training the RF classifier and dimensionality reduction
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split

# The color of the colony
def color_extract(img, gray):
    blur = cv2.blur(gray, (3, 3)) # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_TRIANGLE)
    
    # Finding contours for the thresholded image and use the largest one 
    # to create a mask
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if (len(contours) == 0):
        return 0

    max_ind = np.argmax([cv2.contourArea(cnt) for cnt in contours])

    mask = np.zeros(img.shape, np.uint8)
    cv2.drawContours(mask, [contours[max_ind]], -1, color=(255, 255, 255), thickness=cv2.FILLED)

    # Calculate mean rbg value of pixels inside the mask
    img_mask = img[np.where(mask == 255)]

    img_avg = np.mean(img_mask, axis=0)
    return img_avg

# The roundness of the shape of the colony
def hullness(gray):
    blur = cv2.blur(gray, (3, 3)) # blur the image
    ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_TRIANGLE)
    # Finding contours for the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # create hull array for convex hull points
    hulls = []
    
    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hulls.append(cv2.convexHull(contours[i], False))
    
    # find contour areas
    c_areas = [cv2.contourArea(cnt) for cnt in contours if 20 < cv2.contourArea(cnt)]
    h_areas = [cv2.contourArea(cnt) for cnt in hulls if 20 < cv2.contourArea(cnt)]

    # print('contour areas: ', sorted(c_areas))
    # print('hull areas: ', sorted(h_areas))

    # calculate roundness
    if (len(c_areas) == 0 or len(h_areas) == 0): return 0
    c_area_max = max(c_areas)
    h_area_max = max(h_areas)
    if (c_area_max == 0 or h_area_max == 0): return 0
    return c_area_max/h_area_max

def entropy_g(img_grey):
    # Entroy algorithm
    entropy_img = entropy(img_grey, disk(3))

    # Turn image into binary
    thresh = threshold_otsu(entropy_img)
    binary = entropy_img <= thresh

    # Gaussian blurr
    gau_img = gaussian(binary, sigma = 1)

    # Turn image into binary
    thresh = threshold_otsu(gau_img)
    binary = gau_img <= thresh

    return np.sum(binary == 1) / (np.sum(binary == 1) + np.sum(binary == 0))

def feature_calculation(img_name):
    img = cv2.imread(img_name)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Entropy with Gaussian Blur
    ent = entropy_g(img_grey)

    # Contrast
    contrast = img_grey.std()
    # Entropy with Gaussian Blur
    ent = entropy_g(img_grey)

    # Contrast
    contrast = img_grey.std()

    # Fourier transform
    fft_features = np.fft.fft2(img_grey)
    fft_magnitude = np.abs(np.fft.fftshift(fft_features))
    fft_mean = np.mean(fft_magnitude)
    fft_std = np.std(fft_magnitude)

    # Haralick Texture Features
    glcm = feature.graycomatrix(img_grey, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    haralick_features = [feature.graycoprops(glcm, prop='contrast')[0], feature.graycoprops(glcm, prop='energy')[0],
        feature.graycoprops(glcm, prop='dissimilarity')[0], feature.graycoprops(glcm, prop='homogeneity')[0], 
        feature.graycoprops(glcm, prop='ASM')[0], feature.graycoprops(glcm, prop='correlation')[0]]

    # LBP features
    lbp_features = feature.local_binary_pattern(img_grey, P=8, R=1, method='uniform').ravel()
    lbp_mean = np.mean(lbp_features)
    lbp_std = np.std(lbp_features)

    # Shape Features
    area = measure.regionprops(img_grey)[0].area  # assuming a single connected region
    aspect_ratio = img_grey.shape[1] / img_grey.shape[0]

    # Hullness of the colony shape
    hull = hullness(img_grey)

    # Color of the colony -- average rgb value of pixels inside the colony
    clr = color_extract(img, img_grey)
    
    features = [ent, contrast, fft_mean, fft_std, lbp_mean, lbp_std, area, 
                aspect_ratio] + haralick_features + [hull, clr]
    features_final = [float(val) for val in features]
    
    return features_final