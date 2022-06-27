from __future__ import print_function
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage import data, color, img_as_ubyte
import skimage
import skimage.io
import sklearn
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, accuracy_score, RocCurveDisplay
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
import glob
import os
import pandas as pd
import time
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import disk
import numpy.ma as ma
import random
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import random as rng
import math
import time 
from skimage import exposure
from skimage.filters import threshold_otsu
from skimage.filters import threshold_multiotsu


### reading in image

id_temp = "/Users/benjaminsteinhart/Desktop/image_seg/manual_seg_clean_all/001-011c_pupil_right208.jpg"
# id_temp = "/Users/benjaminsteinhart/Desktop/image_seg/manual_seg_clean_all/001-011c_pupil_right117.jpg" #watershed example

image = cv.imread(id_temp) #for right eye, try right bound at 1600



image = cv.cvtColor(image,cv.COLOR_BGR2GRAY) 

## median filter

img_med = ndi.median_filter(image, size=15)


## log transform 
gamma_value = 0.7

log_image = np.array((255/255**gamma_value)*(img_med**(gamma_value)),dtype='uint8')


### otsu binary

 # ## otsu truncated 
mask = np.logical_and(log_image >=0, log_image <= 255)
# print(mask)
thresh = threshold_otsu(log_image[mask])
print('threshold is',thresh)

# print('min thresh', thresh)
img_bin = np.zeros((log_image.shape))
img_bin[log_image <= thresh] = 255
img_bin = img_bin.astype('int64')


 # ## otsu truncated 
mask = np.logical_and(log_image > 50, log_image < 85)

thresh_mask = np.logical_and(log_image > 5, log_image < 255)

quant1 = 0.01
quant2 = 0.2

ots1 = np.quantile(log_image[thresh_mask].ravel(), quant1)
ots2 = np.quantile(log_image[thresh_mask].ravel(), quant2)
mask = np.logical_and(log_image > ots1, log_image < ots2)

thresholds = threshold_multiotsu(log_image[mask])
print("thresholds", thresholds)


img_bin = np.zeros((log_image.shape))
img_bin[log_image <= thresholds[0]] = 255
img_bin = img_bin.astype('int64')

regions = np.digitize(log_image, bins=thresholds)



img_bin = img_bin.astype('int64')

### morpholoy 

morph_disk = 8

footprint = disk(morph_disk)

closed = closing(img_bin, footprint)


opened = opening(closed, footprint)

#### canny
max_thresh = 220
min_thresh = 100
im_copy = np.uint8(opened)


distance = ndi.distance_transform_edt(opened)
coords = peak_local_max(distance, footprint=np.ones((25, 25)), labels=opened )
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = watershed(-distance, markers, mask=opened )
labels2 = watershed(-distance, markers, mask=opened )


temp_labels = np.unique(labels)
minor_axis_list = []
major_axis_list = []
centerX_list = []
centerY_list = []
rotation_list = []
area_list = []
eccentricity_list = []

    # print(np.unique(labels))

for label in temp_labels:
    img_new = np.zeros((image.shape), dtype = "float64")
    img_new[labels==label] = 255

    im_copy = np.uint8(img_new)
    canny_output = cv.Canny(im_copy, 100, 200, apertureSize = 3)
    contours, hierachy = cv.findContours(canny_output,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        if c.shape[0] > 5:
            minEllipse[i] = cv.fitEllipse(c)

    for i, c in enumerate(contours):
        # ellipse
        if c.shape[0] > 10:
            # area of ellipse
            minor_axis = minEllipse[i][1][0]
            major_axis = minEllipse[i][1][1]
            centerX = minEllipse[i][0][0]
            centerY = minEllipse[i][0][1]
            center = minEllipse[i][0]
            rotation = minEllipse[i][2]
            area = math.pi * major_axis * minor_axis
            #print(area)
            # e = the eccentricity of the ellipse. e2 = 1 - b2/a2.; will assume pupil object is closest to zero
            e = 1 - ((minor_axis/2)**2)/((major_axis/2)**2)
            #print(e)
            if area > 5000 and e < 0.5:
                if area <100000:

                    #print("the minor axis is " , minor_axis)
                    #print("the major axis is ",major_axis)
                    minor_axis_list.append(minor_axis)
                    major_axis_list.append(major_axis)
                    centerX_list.append(centerX)
                    centerY_list.append(centerY)
                    rotation_list.append(rotation)
                    area_list.append(area)
                    eccentricity_list.append(e)
ecc_min = 100
index_fin = 0
for i in range(len(eccentricity_list)):
    if eccentricity_list[i] < ecc_min:
        index_fin = i
        ecc_min = eccentricity_list[i]

img_new = np.zeros((image.shape), dtype = "float64")
img_new[labels==temp_labels[4]] = 255
# im_copy = np.uint8(img_new)
im_copy = np.uint8(opened)

canny_output = cv.Canny(im_copy, min_thresh, max_thresh, apertureSize = 3)


# canny_output = cv.Canny(im_copy, min_thresh, max_thresh, apertureSize = 3)
def thresh_callback(val):

    color1 = 	(0,255,255)
    color2 = 	(127,255,0)
    color3 = (191,62,255)
    color4 = (255,20,147)
    color5 = (255,255,0)
    color6 = (255,250,250)
    colors = (color1, color2, color3, color4, color5, color6)
    contours, hierachy= cv.findContours(canny_output,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    # Find the rotated rectangles and ellipses for each contour
    minRect = [None]*len(contours)
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv.fitEllipse(c)
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    for i, c in enumerate(contours):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # contour
        cv.drawContours(drawing, contours, i, colors[i])
        # ellipse
        if c.shape[0] > 5:
            cv.ellipse(drawing, minEllipse[i], colors[i], 2)
        # rotated rectangle
        box = cv.boxPoints(minRect[i])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv.drawContours(drawing, [box], 0, color)
        print("i is ", i)



    # sum_tot = sum(minor_axis_list)

    # print(sum_tot)

    #return major_fin, minor_fin
    cv.imshow('Contours', drawing)
    return contours, drawing


parser = argparse.ArgumentParser(description='Code for Creating Bounding rotated boxes and ellipses for contours tutorial.')
parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
args = parser.parse_args()

src_gray = image

source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src_gray)
max_thresh = 255
thresh = 25 # initial threshold - half of 255, implies probability of being a 1 (pupil) is 50% or greater 
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
contours, drawing = thresh_callback(thresh)
cv.waitKey()



### Function that puts ellipses on original image
def image_overlap(image, center, axis_list, rotation ):

    img_shape = image.shape

    # black_image = np.zeros(img_shape)

    center_coordinates = (center)

    axesLength = (axis_list)

    angle = rotation

    startAngle = 0

    endAngle = 360

    # Blue color in BGR
    color = (255, 255, 255)

    # Line thickness of -1 px
    thickness = 3

    # Using cv2.ellipse() method
    # Draw a ellipse with blue line borders of thickness of -1 px
    image = cv.ellipse(image, center_coordinates, axesLength, angle,
                            startAngle, endAngle, color, thickness)


    return image

## prediction for mask 
def mask_prediction(image, center, axis_list, rotation ):

    img_shape = image.shape

    black_image = np.zeros(img_shape)

    center_coordinates = (center)

    axesLength = (axis_list)

    angle = rotation

    startAngle = 0

    endAngle = 360

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of -1 px
    thickness = -1

    # Using cv2.ellipse() method
    # Draw a ellipse with blue line borders of thickness of -1 px
    image = cv.ellipse(black_image, center_coordinates, axesLength, angle,
                            startAngle, endAngle, color, thickness)

    return image

centerX_list[index_fin]
centerY_list[index_fin]
minor_axis_list[index_fin]
major_axis_list[index_fin]
rotation_list[index_fin]
eccentricity_list[index_fin] 

# im_overlap = image_overlap(cv.imread(file), (round(centerX[i]), round(centerY[i])), (round(minor_axis[i]/2), round(major_axis[i]/2)), round(rotation[i]))


ellipse_filtered = image_overlap(np.zeros((image.shape)), (round(centerX_list[index_fin]), round(centerY_list[index_fin])), (round(minor_axis_list[index_fin]/2), round(major_axis_list[index_fin]/2)), round(rotation_list[index_fin]))


img_fin = image_overlap(image, (round(centerX_list[index_fin]), round(centerY_list[index_fin])), (round(minor_axis_list[index_fin]/2), round(major_axis_list[index_fin]/2)), round(rotation_list[index_fin]))

img_mask = mask_prediction(image, (round(centerX_list[index_fin]), round(centerY_list[index_fin])), (round(minor_axis_list[index_fin]/2), round(major_axis_list[index_fin]/2)), round(rotation_list[index_fin]))
im_copy = np.uint8(opened)

# canny_output = cv.Canny(im_copy, min_thresh, max_thresh, apertureSize = 3)

###### Figure 3 ####

# fig, axes = plt.subplots(ncols=7, figsize=(9, 3))
# ax = axes.ravel()


original = cv.imread(id_temp) #for right eye, try right bound at 1600

original = cv.cvtColor(original,cv.COLOR_BGR2GRAY) 

### Piece for Figure 1

## with 6 panels ; abreviated process 

# fig, axes = plt.subplots(nrows = 2, ncols =  3)

# axes[0,0].imshow(original, cmap="gray")
# axes[0,0].set_title('1') # Original
# axes[0,1].imshow(log_image, cmap="gray")
# axes[0,1].set_title('2') # median + intensity transform
# axes[0,2].imshow(regions, cmap="gray")
# axes[0,2].set_title('3') # Otsu multi thresh
# axes[1,0].imshow(opened, cmap="gray")
# axes[1,0].set_title('4') # Post morphology + watershed identification
# axes[1,1].imshow(drawing, cmap="gray")
# axes[1,1].set_title('5') # canny edge output
# axes[1,2].imshow(img_fin, cmap="gray")
# axes[1,2].set_title('6') # ellipse on original

# axes[0,0].set_axis_off()
# axes[0,1].set_axis_off()
# axes[0,2].set_axis_off()
# axes[1,0].set_axis_off()
# axes[1,1].set_axis_off()
# axes[1,2].set_axis_off()

# fig.tight_layout()
# plt.show()


#############################################
########## Overview of Entire Process #######
#############################################


# kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
# dilate = cv.dilate(canny_output, kernel, iterations=1)

# fig, axes = plt.subplots(nrows = 3,ncols =  5)


# axes[0,0].imshow(original, cmap="gray")
# axes[0,0].set_title('A') # Original
# axes[0,1].imshow(img_med, cmap="gray")
# axes[0,1].set_title('B') # Median Filter
# axes[0,2].imshow(log_image, cmap="gray")
# axes[0,2].set_title('C') # Intensity Transform
# axes[0,3].hist(log_image.flatten(), bins=256, range=(0, 255))
# axes[0,3].axvline(x=thresholds[0], color='r', linestyle='dashed', linewidth=2)
# axes[0,3].axvline(x=thresholds[1], color='b', linestyle='dashed', linewidth=2)
# axes[0,3].axvline(x=ots1, color='y', linestyle='dashed', linewidth=1)
# axes[0,3].axvline(x=ots2, color='y', linestyle='dashed', linewidth=1)
# axes[0,3].set_title('D') # Histogram
# axes[0,4].imshow(regions, cmap="gray")
# axes[0,4].set_title('E') # Otsu Multi-Threshold Binarization
# axes[1,0].imshow(img_bin, cmap="gray")
# axes[1,0].set_title('F') # Binarization of Pupil
# axes[1,1].imshow(closed, cmap="gray")
# axes[1,1].set_title('G') # Closing Image Morphology
# axes[1,2].imshow(opened, cmap="gray")
# axes[1,2].set_title('H') # Opening Image Morphology
# axes[1,3].imshow(dilate, cmap="gray")
# axes[1,3].set_title('I') # Edge Detection
# axes[1,4].imshow(drawing, cmap="gray")
# axes[1,4].set_title('J') # Ellipses
# axes[2,0].imshow(ellipse_filtered, cmap="gray")
# axes[2,0].set_title('K') # Ellipses filtered 
# axes[2,1].imshow(img_mask, cmap="gray")
# axes[2,1].set_title('L') # Binary Prediction
# axes[2,2].imshow(img_fin, cmap="gray")
# axes[2,2].set_title('M') # Prediction on Original


# axes[0,0].set_axis_off()
# axes[0,1].set_axis_off()
# axes[0,2].set_axis_off()
# axes[0,3].set_axis_off()
# axes[0,4].set_axis_off()
# axes[1,0].set_axis_off()
# axes[1,1].set_axis_off()
# axes[1,2].set_axis_off()
# axes[1,3].set_axis_off()
# axes[1,4].set_axis_off()
# axes[2,0].set_axis_off()
# axes[2,1].set_axis_off()
# axes[2,2].set_axis_off()
# axes[2,3].set_axis_off()
# axes[2,4].set_axis_off()

# # for a in ax:
# #     a.set_axis_off()

# fig.tight_layout()
# plt.show()

#############################################
########## Just image and mask for fig 7 overview #######
#############################################
# fig, axes = plt.subplots(nrows = 1,ncols =  2)


# axes[0].imshow(original, cmap="gray")
# axes[1].imshow(img_mask, cmap="gray")

# axes[0].set_axis_off()
# axes[1].set_axis_off()

# # for a in ax:
# #     a.set_axis_off()

# fig.tight_layout()
# plt.show()

###################################
########## Figure #######
###################################


###################################
########## Watershed Figure #######
###################################


### reading in image

# temp = "/Users/benjaminsteinhart/Desktop/image_seg/manual_seg_clean_all/001-011c_pupil_right117.jpg"
# image = cv.imread(temp) #for right eye, try right bound at 1600

# image = cv.cvtColor(image,cv.COLOR_BGR2GRAY) 

# ## median filter

# img_med = ndi.median_filter(image, size=15)


# ## log transform 
# gamma_value = 0.7

# log_image = np.array((255/255**gamma_value)*(img_med**(gamma_value)),dtype='uint8')


# ### otsu binary

#  # ## otsu truncated 
# mask = np.logical_and(log_image >=0, log_image <= 255)
# # print(mask)
# thresh = threshold_otsu(log_image[mask])
# print('threshold is',thresh)

# # print('min thresh', thresh)
# img_bin = np.zeros((log_image.shape))
# img_bin[log_image <= thresh] = 255
# img_bin = img_bin.astype('int64')


#  # ## otsu truncated 
# mask = np.logical_and(log_image > 50, log_image < 85)

# thresh_mask = np.logical_and(log_image > 5, log_image < 255)

# quant1 = 0.01
# quant2 = 0.2

# ots1 = np.quantile(log_image[thresh_mask].ravel(), quant1)
# ots2 = np.quantile(log_image[thresh_mask].ravel(), quant2)
# mask = np.logical_and(log_image > ots1, log_image < ots2)

# thresholds = threshold_multiotsu(log_image[mask])
# print("thresholds", thresholds)


# img_bin = np.zeros((log_image.shape))
# img_bin[log_image <= thresholds[0]] = 255
# img_bin = img_bin.astype('int64')

# regions = np.digitize(log_image, bins=thresholds)


# img_bin = img_bin.astype('int64')

# ### morpholoy 

# morph_disk = 8

# footprint = disk(morph_disk)

# closed = closing(img_bin, footprint)


# opened = opening(closed, footprint)

# #### canny
# max_thresh = 220
# min_thresh = 100
# im_copy = np.uint8(opened)

# canny_output = cv.Canny(im_copy, min_thresh, max_thresh, apertureSize = 3)


# kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
# dilate = cv.dilate(canny_output, kernel, iterations=1)


# distance = ndi.distance_transform_edt(opened)
# coords = peak_local_max(distance, footprint=np.ones((25, 25)), labels=opened )
# mask = np.zeros(distance.shape, dtype=bool)
# mask[tuple(coords.T)] = True
# markers, _ = ndi.label(mask)
# labels = watershed(-distance, markers, mask=opened )

# temp_labels = np.unique(labels)
# minor_axis_list = []
# major_axis_list = []
# centerX_list = []
# centerY_list = []
# rotation_list = []
# area_list = []
# eccentricity_list = []


# ##### Pre watershed poor fit 

# contours, hierachy = cv.findContours(canny_output,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

# minEllipse = [None]*len(contours)
# for i, c in enumerate(contours):
#     if c.shape[0] > 5:
#         minEllipse[i] = cv.fitEllipse(c)

# for i, c in enumerate(contours):
#     # ellipse
#     if c.shape[0] > 10:
#         # area of ellipse
#         minor_axis = minEllipse[i][1][0]
#         major_axis = minEllipse[i][1][1]
#         centerX = minEllipse[i][0][0]
#         centerY = minEllipse[i][0][1]
#         center = minEllipse[i][0]
#         rotation = minEllipse[i][2]
#         area = math.pi * major_axis * minor_axis
#         #print(area)
#         # e = the eccentricity of the ellipse. e2 = 1 - b2/a2.; will assume pupil object is closest to zero
#         e = 1 - ((minor_axis/2)**2)/((major_axis/2)**2)
#         #print(e)
#         if area > 5000 and e < 0.5:
#             if area <100000:

#                 #print("the minor axis is " , minor_axis)
#                 #print("the major axis is ",major_axis)
#                 minor_axis_list.append(minor_axis)
#                 major_axis_list.append(major_axis)
#                 centerX_list.append(centerX)
#                 centerY_list.append(centerY)
#                 rotation_list.append(rotation)
#                 area_list.append(area)
#                 eccentricity_list.append(e)
# ecc_min = 100
# index_fin = 0
# for i in range(len(eccentricity_list)):
#     if eccentricity_list[i] < ecc_min:
#         index_fin = i
#         ecc_min = eccentricity_list[i]

# im_copy2 = np.uint8(image)

# img_fin_poor = image_overlap(image, (round(centerX_list[index_fin]), round(centerY_list[index_fin])), (round(minor_axis_list[index_fin]/2), round(major_axis_list[index_fin]/2)), round(rotation_list[index_fin]))

# ###### Watershed results

# minor_axis_list = []
# major_axis_list = []
# centerX_list = []
# centerY_list = []
# rotation_list = []
# area_list = []
# eccentricity_list = []
# img_new = np.zeros((image.shape), dtype = "float64")
# img_new[labels==temp_labels[7]] = 255 # for component piece 

# im_copy = np.uint8(img_new)
# canny_output = cv.Canny(im_copy, 100, 200, apertureSize = 3)

# kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
# dilate = cv.dilate(canny_output, kernel, iterations=1)


# contours, hierachy = cv.findContours(canny_output,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

# minEllipse = [None]*len(contours)
# for i, c in enumerate(contours):
#     if c.shape[0] > 5:
#         minEllipse[i] = cv.fitEllipse(c)

# for i, c in enumerate(contours):
#     # ellipse
#     if c.shape[0] > 10:
#         # area of ellipse
#         minor_axis = minEllipse[i][1][0]
#         major_axis = minEllipse[i][1][1]
#         centerX = minEllipse[i][0][0]
#         centerY = minEllipse[i][0][1]
#         center = minEllipse[i][0]
#         rotation = minEllipse[i][2]
#         area = math.pi * major_axis * minor_axis
#         #print(area)
#         # e = the eccentricity of the ellipse. e2 = 1 - b2/a2.; will assume pupil object is closest to zero
#         e = 1 - ((minor_axis/2)**2)/((major_axis/2)**2)
#         #print(e)
#         if area > 5000 and e < 0.5:
#             if area <100000:

#                 #print("the minor axis is " , minor_axis)
#                 #print("the major axis is ",major_axis)
#                 minor_axis_list.append(minor_axis)
#                 major_axis_list.append(major_axis)
#                 centerX_list.append(centerX)
#                 centerY_list.append(centerY)
#                 rotation_list.append(rotation)
#                 area_list.append(area)
#                 eccentricity_list.append(e)
# ecc_min = 100
# index_fin = 0
# for i in range(len(eccentricity_list)):
#     if eccentricity_list[i] < ecc_min:
#         index_fin = i
#         ecc_min = eccentricity_list[i]

# ellipse_filtered = image_overlap(np.zeros((image.shape)), (round(centerX_list[index_fin]), round(centerY_list[index_fin])), (round(minor_axis_list[index_fin]/2), round(major_axis_list[index_fin]/2)), round(rotation_list[index_fin]))

# image = cv.imread(temp) #for right eye, try right bound at 1600

# image = cv.cvtColor(image,cv.COLOR_BGR2GRAY) 

# img_fin = image_overlap(image, (round(centerX_list[index_fin]), round(centerY_list[index_fin])), (round(minor_axis_list[index_fin]/2), round(major_axis_list[index_fin]/2)), round(rotation_list[index_fin]))

# img_mask = mask_prediction(image, (round(centerX_list[index_fin]), round(centerY_list[index_fin])), (round(minor_axis_list[index_fin]/2), round(major_axis_list[index_fin]/2)), round(rotation_list[index_fin]))

# image = cv.imread(temp) #for right eye, try right bound at 1600

# image = cv.cvtColor(image,cv.COLOR_BGR2GRAY) 

# fig, axes = plt.subplots(2, 4)

# axes[0,0].imshow(img_fin_poor, cmap="gray")
# axes[0,0].set_title('A') # Original
# axes[0,1].imshow(opened, cmap="gray")
# axes[0,1].set_title('B') # binary image post image morphology
# axes[0,2].imshow(labels, cmap="gray")
# axes[0,2].set_title('C') # Watershed Labels
# axes[0,3].imshow(img_new, cmap="gray")
# axes[0,3].set_title('D') # Watershed Pupil Label
# axes[1,0].imshow(dilate, cmap="gray")
# axes[1,0].set_title('E') # Edges from pupil label
# axes[1,1].imshow(ellipse_filtered, cmap="gray")
# axes[1,1].set_title('F') # ellipse from edges
# axes[1,2].imshow(img_mask, cmap="gray") 
# axes[1,2].set_title('G') # binary mask
# axes[1,3].imshow(img_fin, cmap="gray")
# axes[1,3].set_title('H') # ellipse on image

# axes[0,0].set_axis_off()
# axes[0,1].set_axis_off()
# axes[0,2].set_axis_off()
# axes[0,3].set_axis_off()
# axes[1,0].set_axis_off()
# axes[1,1].set_axis_off()
# axes[1,2].set_axis_off()
# axes[1,3].set_axis_off()
# fig.tight_layout()
# plt.show()


###### Thesis Presentation Figures 

## crash course 

# id_temp = "/Users/benjaminsteinhart/Desktop/image_seg/reports/figures_thesis/Lola.jpg"

# # image = cv.imread("/Users/benjaminsteinhart/Desktop/image_seg/manual_segment_clean/participant55/001-037c_pupil_left260.jpg") #for right eye, try right bound at 1600
# image = cv.imread(id_temp) #for right eye, try right bound at 1600
# # image = cv.imread("/Users/benjaminsteinhart/Desktop/image_seg/manual_seg_clean_all/001-017a_pupil_right26.jpg") #for right eye, try right bound at 1600


# dog_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY) 

# fig, axes = plt.subplots(nrows = 1,ncols =  2)


# axes[0].imshow(dog_image, cmap="gray")
# axes[1].hist(dog_image.flatten(), bins=256, range=(0, 255))


# axes[0].set_axis_off()

# fig.tight_layout()
# plt.show()

## Median image


id_temp = "/Users/benjaminsteinhart/Desktop/image_seg/manual_seg_clean_all/001-011c_pupil_right208.jpg"

image = cv.imread(id_temp) #for right eye, try right bound at 1600


image = cv.cvtColor(image,cv.COLOR_BGR2GRAY) 

# fig, axes = plt.subplots(nrows = 1,ncols =  2)


# axes[0].imshow(image, cmap="gray")
# axes[0].set_title('Original') # Original

# axes[1].imshow(img_med, cmap="gray")
# axes[1].set_title('Median Filtered') # Original


# axes[0].set_axis_off()
# axes[1].set_axis_off()


# # for a in ax:
# #     a.set_axis_off()

# fig.tight_layout()
# plt.show()


#### Power-law intensity transform 

# gamma_value = 2.0

# log_image_dark = np.array((255/(255**gamma_value))*(img_med**(gamma_value)),dtype='float')*255


# gamma_value = .7

# log_image_light = np.array((255/255**gamma_value)*(img_med**(gamma_value)),dtype='uint8')


# fig, axes = plt.subplots(nrows = 1,ncols =  3)


# axes[0].imshow(log_image_dark, cmap="gray")
# axes[1].imshow(log_image_light, cmap="gray")
# axes[2].imshow(img_med, cmap="gray")


# axes[0].set_axis_off()
# axes[1].set_axis_off()
# axes[2].set_axis_off()


# # for a in ax:
# #     a.set_axis_off()

# fig.tight_layout()
# plt.show()

## Binarization

# thresh1 = np.mean(log_image_light)
# thresh2 = np.quantile(log_image_light, 0.25)
# thresh3 = np.quantile(log_image_light, 0.1)



# threshes = (thresh1, thresh2, thresh3)

# names = ("bin1", "bin2", "bin3")
# namecount = 0
# for thresh in threshes:

#     img_bin = np.zeros((log_image_light.shape))
#     img_bin[log_image_light <= thresh] = 255
#     img_bin = img_bin.astype('int64')


#     fig, axes = plt.subplots(nrows = 1,ncols =  3, figsize=(8,3))

#     axes[0].imshow(log_image_light, cmap="gray")
#     axes[0].set_title('Intensity Transformed') # Original
#     axes[1].hist(log_image_light.flatten(), bins=256, range=(0, 255))
#     axes[1].axvline(thresh, color='red', linestyle='dashed', linewidth=1)
#     axes[1].set_title('Histogram') # Original
#     axes[2].imshow(img_bin, cmap="gray")
#     axes[2].set_title('Binarized Image') # Original


#     axes[0].set_axis_off()
#     axes[2].set_axis_off()

#     fig.tight_layout()
#     plt.savefig("/Users/benjaminsteinhart/Desktop/image_seg/reports/figures_thesis/" + names[namecount] +".jpg")
#     namecount +=1


# Otsu's binarization 

### otsu binary

 # ## otsu truncated 
# print(mask)
# thresh = threshold_otsu(log_image)
# print('threshold is',thresh)

# # print('min thresh', thresh)
# img_bin = np.zeros((log_image_light.shape))
# img_bin[log_image_light <= thresh] = 255
# img_bin = img_bin.astype('int64')



# fig, axes = plt.subplots(nrows = 1,ncols =  3, figsize=(8,3))

# axes[0].imshow(log_image_light, cmap="gray")
# axes[0].set_title('Intensity Transformed') # Original
# axes[1].hist(log_image_light.flatten(), bins=256, range=(0, 255))
# axes[1].axvline(thresh, color='red', linestyle='dashed', linewidth=1)
# axes[1].set_title('Histogram') # Original
# axes[2].imshow(img_bin, cmap="gray")
# axes[2].set_title('Binarized Image') # Original

# axes[0].set_axis_off()
# axes[2].set_axis_off()

# fig.tight_layout()
# plt.savefig("/Users/benjaminsteinhart/Desktop/image_seg/reports/figures_thesis/otsu_k2.jpg")

#### Otsu multi-class not truncated #####

# thresholds = threshold_multiotsu(log_image_light)
# print("thresholds", thresholds)


# img_bin = np.zeros((log_image.shape))
# img_bin[log_image <= thresholds[0]] = 255
# img_bin = img_bin.astype('int64')

# regions = np.digitize(log_image, bins=thresholds)


# img_bin = img_bin.astype('int64')



# fig, axes = plt.subplots(nrows = 1,ncols =  3, figsize=(8,3))

# axes[0].imshow(log_image_light, cmap="gray")
# axes[0].set_title('Intensity Transformed') # Original
# axes[1].hist(log_image_light.flatten(), bins=256, range=(0, 255))
# axes[1].axvline(thresholds[0], color='red', linestyle='dashed', linewidth=1)
# axes[1].axvline(thresholds[1], color='blue', linestyle='dashed', linewidth=1)
# axes[1].set_title('Histogram') # Original
# axes[2].imshow(regions, cmap="gray")
# axes[2].set_title('Three Class Image') # Original

# axes[0].set_axis_off()
# axes[2].set_axis_off()

# fig.tight_layout()
# plt.savefig("/Users/benjaminsteinhart/Desktop/image_seg/reports/figures_thesis/otsu_k3.jpg")



#### Otsu multi-class not truncated #####

# quant1 = 0.01
# quant2 = 0.2

# ots1 = np.quantile(log_image_light[thresh_mask].ravel(), quant1)
# ots2 = np.quantile(log_image_light[thresh_mask].ravel(), quant2)
# mask = np.logical_and(log_image_light > ots1, log_image_light < ots2)

# thresholds = threshold_multiotsu(log_image_light[mask])
# print("thresholds", thresholds)


# img_bin = np.zeros((log_image_light.shape))
# img_bin[log_image_light <= thresholds[0]] = 255
# img_bin = img_bin.astype('int64')

# regions = np.digitize(log_image_light, bins=thresholds)


# img_bin = img_bin.astype('int64')



fig, axes = plt.subplots(nrows = 1,ncols =  3, figsize=(8,3))

axes[0].imshow(log_image, cmap="gray")
axes[0].set_title('Intensity Transformed') # Original
axes[1].hist(log_image.flatten(), bins=256, range=(0, 255))
axes[1].axvline(thresholds[0], color='red', linestyle='dashed', linewidth=1)
axes[1].axvline(thresholds[1], color='red', linestyle='dashed', linewidth=1)
axes[1].axvline(ots1, color='green', linestyle='dashed', linewidth=1)
axes[1].axvline(ots2, color='green', linestyle='dashed', linewidth=1)

axes[1].set_title('Histogram') # Original
axes[2].imshow(regions, cmap="gray")
axes[2].set_title('Three Class Image') # Original

axes[0].set_axis_off()
axes[2].set_axis_off()

fig.tight_layout()

plt.savefig("/Users/benjaminsteinhart/Desktop/image_seg/reports/figures_thesis/otsu_k3_truncated.jpg")


##### Image Morphology
### morpholoy 

morph_disk = 8
footprint = disk(morph_disk)

closed = closing(img_bin, footprint)


opened = opening(closed, footprint)

im_copy = np.uint8(opened)


# fig, axes = plt.subplots(nrows = 1,ncols =  3, figsize=(8,3))

# axes[0].imshow(img_bin, cmap="gray")
# axes[0].set_title('Binary Image') # Original
# axes[1].imshow(closed, cmap="gray")
# axes[1].set_title('Closing') # Originalaxes[2].imshow(regions, cmap="gray")
# axes[2].imshow(opened, cmap="gray")
# axes[2].set_title('Opening') # Originalaxes[2].imshow(regions, cmap="gray")

# axes[0].set_axis_off()
# axes[1].set_axis_off()

# axes[2].set_axis_off()

# fig.tight_layout()
# plt.savefig("/Users/benjaminsteinhart/Desktop/image_seg/reports/figures_thesis/morphology.jpg")


### Canny Edge Detection Results

canny_output = cv.Canny(im_copy, min_thresh, max_thresh, apertureSize = 3)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
dilate = cv.dilate(canny_output, kernel, iterations=1)


# fig, axes = plt.subplots(nrows = 1,ncols =  2, figsize=(8,3))

# axes[0].imshow(opened, cmap="gray")
# axes[0].set_title('Post Image Morphology') # Original
# axes[1].imshow(dilate, cmap="gray")
# axes[1].set_title('Edge Detection') # Original

# axes[0].set_axis_off()
# axes[1].set_axis_off()

# fig.tight_layout()
# plt.savefig("/Users/benjaminsteinhart/Desktop/image_seg/reports/figures_thesis/edges.jpg")
# plt.show()

#### Ellipse Output

# axes[1,4].imshow(drawing, cmap="gray")
# axes[1,4].set_title('J') # Ellipses

# fig, axes = plt.subplots(nrows = 1,ncols =  2, figsize=(8,3))


# axes[0].imshow(dilate, cmap="gray")
# axes[0].set_title('Edge Detection') # edges
# axes[1].imshow(drawing, cmap="gray")
# axes[1].set_title('Ellipses') # Original

# axes[0].set_axis_off()
# axes[1].set_axis_off()

# fig.tight_layout()
# plt.savefig("/Users/benjaminsteinhart/Desktop/image_seg/reports/figures_thesis/multiple_ellipses.jpg")


## Ellipses Filtered

# fig, axes = plt.subplots(nrows = 1,ncols =  2, figsize=(8,3))


# axes[0].imshow(drawing, cmap="gray")
# axes[0].set_title('Ellipses') # edges
# axes[1].imshow(ellipse_filtered, cmap="gray")
# axes[1].set_title('Ellipses Filtered') # Original

# axes[0].set_axis_off()
# axes[1].set_axis_off()

# fig.tight_layout()
# plt.savefig("/Users/benjaminsteinhart/Desktop/image_seg/reports/figures_thesis/filtered_ellipses.jpg")


### Ellipse Output

## Ellipses Filtered

# fig, axes = plt.subplots(nrows = 1,ncols =  3, figsize=(8,3))


# axes[0].imshow(ellipse_filtered, cmap="gray")
# axes[0].set_title('Ellipses Filtered') # edges
# axes[1].imshow(img_mask, cmap="gray")
# axes[1].set_title('Binary Mask') # Original
# axes[2].imshow(img_fin, cmap="gray")
# axes[2].set_title('Ellipse on Original') # Original

# axes[0].set_axis_off()
# axes[1].set_axis_off()
# axes[2].set_axis_off()


# fig.tight_layout()
# plt.savefig("/Users/benjaminsteinhart/Desktop/image_seg/reports/figures_thesis/ellipse_output.jpg")


## Watershed example ## 

# fig, axes = plt.subplots(nrows = 1,ncols =  4, figsize=(8,3))


# axes[0].imshow(original, cmap="gray")
# axes[0].set_title('Original') # Original
# axes[1].imshow(opened, cmap="gray")
# axes[1].set_title('Post Image Morphology') # Opening Image Morphology
# axes[2].imshow(dilate, cmap="gray")
# axes[2].set_title('Edges') # Edge Detection
# axes[3].imshow(img_fin, cmap="gray")
# axes[3].set_title('Ellipse on Original') # Prediction on Original


# axes[0].set_axis_off()
# axes[1].set_axis_off()
# axes[2].set_axis_off()
# axes[3].set_axis_off()


# # for a in ax:
# #     a.set_axis_off()

# fig.tight_layout()

# plt.savefig("/Users/benjaminsteinhart/Desktop/image_seg/reports/figures_thesis/watershed_failure.jpg")


##### Pic of eyes


# #### image eye 1 
# id_temp = "/Users/benjaminsteinhart/Desktop/image_seg/manual_segmentation/original_frames/participant16/001-005a_pupil_left26.jpg" 

# image1 = cv.imread(id_temp) #for right eye, try right bound at 1600
# image1 = cv.cvtColor(image1,cv.COLOR_BGR2GRAY) 


# ### image eye 2 
# id_temp = "/Users/benjaminsteinhart/Desktop/image_seg/manual_segmentation/original_frames/participant36/001-047b_pupil_left52.jpg" 

# image2 = cv.imread(id_temp) #for right eye, try right bound at 1600
# image2 = cv.cvtColor(image2,cv.COLOR_BGR2GRAY) 


#### image eye 3 

### image eye closed


# id_temp = "/Users/benjaminsteinhart/Desktop/image_seg/manual_segmentation/original_frames/participant8/001-068a_pupil_right39.jpg" #watershed example

# image4 = cv.imread(id_temp) #for right eye, try right bound at 1600
# image4 = cv.cvtColor(image4,cv.COLOR_BGR2GRAY) 


# fig, axes = plt.subplots(nrows = 2,ncols =  2)


# axes[0,0].imshow(image1, cmap="gray")
# axes[0,0].set_title('A') # Original
# axes[0,1].imshow(image2, cmap="gray")
# axes[0,1].set_title('B') # Opening Image Morphology
# axes[1,0].imshow(image3, cmap="gray")
# axes[1,0].set_title('C') # Edge Detection
# axes[1,1].imshow(image4, cmap="gray")
# axes[1,1].set_title('D') # Prediction on Original


# axes[0,0].set_axis_off()
# axes[0,1].set_axis_off()
# axes[1,0].set_axis_off()
# axes[1,1].set_axis_off()


# # for a in ax:
# #     a.set_axis_off()

# fig.tight_layout()

# plt.savefig("/Users/benjaminsteinhart/Desktop/image_seg/reports/figures_thesis/eye_examples.jpg")

