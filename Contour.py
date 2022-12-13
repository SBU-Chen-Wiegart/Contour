# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:36:14 2022

@author: 93969
"""

import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
import os
import glob

path = r'/home/karenchen-wiegart/ChenWiegartgroup/Xiaoyin/Contour'
os.chdir(path)
def hull_convex(img_path):
    img_path = r'109048_alignwith109038_resampled_z360_1.tiff'
    # img_path = 'test.tiff'
    # img = io.imread('D:\\Research_data\\contour\\109048_alignwith109038_resampled_z360.tif')
    img = io.imread(img_path)
    thresh = threshold_otsu(img)
    ret, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    # plt.figure()
    # plt.imshow(binary)
    
    contours, hierarchy = cv2.findContours(image=binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # draw contours on the original image
    img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image_copy =  img_norm.copy()
    
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    
    # see the results
    cv2.imshow('None approximation', image_copy)
    cv2.waitKey(0)
    # cv2.imwrite('contours_none_image1.jpg', image_copy)
    # cv2.destroyAllWindows()
    # plt.imshow(image_copy-img_norm)
    
    # create hull array for convex hull points
    hull = []
    
    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))
        
    # create an empty black image
    drawing = np.zeros((binary.shape[0], binary.shape[1]), np.uint8)
    
    # draw contours and hull points
    for i in range(len(contours)):
        # i = max_idx
        color_contours = (0, 255, 0) # green - color for contours
        color = (255, 0, 0) # blue - color for convex hull
        # draw ith contour
        # cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)
        
    cv2.imshow('hull convex', drawing)
    
    # plt.figure()
    # plt.imshow(drawing+img_norm)
    
    # Find max contours
    max_idx = 0
    for i, contour in enumerate(contours):
        if len(contour)>len(contours[max_idx]):
            max_idx = i
    return hull, max_idx

# names = glob.glob(r'D:\Research_data\contour\*.tif')
# hull = []
# max_idx = []
# for name in names:
#     hull1, max_idx1 = hull_convex(name)

hull1, max_idx1 = hull_convex('109048_alignwith109038_resampled_z360.tiff')
hull2, max_idx2 = hull_convex('109120_alignwith109109_resampled_z360.tiff')

# q = contours[max_idx]
# q = q.squeeze()
# plt.figure()
# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# plt.scatter(q[:, 0], q[:, 1])
# img = cv2.imread('109048_alignwith109038_resampled_z360.tif', 0)
# plt.figure()
# plt.imshow(img)

# Draw max contour
binary = io.imread('109120_alignwith109109_resampled_z360.tif')
drawing = np.zeros((binary.shape[0], binary.shape[1]), np.uint8)
cv2.drawContours(drawing, hull1, max_idx1, color, 1, 8)
cv2.drawContours(drawing, hull2, max_idx2, color, 1, 8)

plt.figure()
plt.imshow(drawing)
