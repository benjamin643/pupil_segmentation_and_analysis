# This document is to gather the following metrics for each image that we have masks for: accuracy, jaccard, f1, precision. 
# outputs each metric for each image into a csv. 

## This document is taking the fitting the actual masks with the hough transform.

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


## Hough

# def mask_prediction(image_path, threshold):

#     image = cv.imread(file)
#     img = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

#     bin2 = img
#     bin2[bin2<threshold] = 0
#     bin2[bin2>0] = 255

#     footprint = disk(10)

#     # plt.imshow(closed)
#     # plt.show()

#     opened = opening(bin2, footprint)

#     closed = closing(opened, footprint)

#     # plt.figure()
#     # plt.imshow(bin2)
#     # plt.show()

#     #edge detection
#     edges = cv.Canny(closed,0,40, apertureSize = 3)

#     # plt.figure()
#     # plt.imshow(edges)
#     # plt.show()

#     # # Detect two radii
#     hough_radii = np.arange(20, 90, 2)
#     hough_res = hough_circle(edges, hough_radii)


#     # Select the most prominent 3 circles
# # Select the most prominent 3 circles
#     accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
#                                             total_num_peaks=1)

#     # Draw them

#     for center_y, center_x, radius in zip(cy, cx, radii):
#         circy, circx = circle_perimeter(center_y, center_x, radius,
#                                         shape=image.shape)
#         image[circy, circx] = (220, 20, 20)
#     # print(radius)
#     # ax.imshow(image, cmap=plt.cm.gray)
#     # plt.show()
#     center = (center_x, center_y)
    

#     img_shape = edges.shape

#     black_image = np.zeros(img_shape)

#     radius = radius

#     # Blue color in BGR
#     color_mask = (255, 0, 0)

#     # Line thickness of -1 px
#     thickness = -1

#     # Using cv2.ellipse() method
#     # Draw a ellipse with blue line borders of thickness of -1 px
#     mask_pred = cv.circle(black_image, center, radius, color_mask, thickness)


#     return mask_pred, radius


# # Note; Not being used to train but to check accuracy of images that were not used in training. 
# def label_extraction(labeled_img):
#      ######################################
#     #Now, add a column in the data frame for the Labels
#     #changing mask to strict binary
#     labeled_img[labeled_img<50] = 0
#     labeled_img[labeled_img>0] = 255
#     #reshape to one column
#     # labeled_img1 = labeled_img.reshape(-1)
#     #########################################################
#     #Define the dependent variable that needs to be predicted (labels)
#     return labeled_img

# #### Due to different naming systems, the sorted function does not correctly line up 
# # the masks and the original images. The below function makes a new list for the masks
# # that modifies the basename, cutting off 'pupil_mask', and is then correctly matched with 
# # the original files using the sorted function. 


# ## matching format of masks (needed to use glob.glob instead of os.list because os.list creates a DS store file)


# acc_score = []
# prec_score = []
# f1_score_list = []
# jaccard_list = []
# time_list = []
# file_names = []
# all_labels = []

# all_predictions = []

# radius_list = []
# ########

# ### Creating dictionary of true mask labels
# path_to_masks = "/Users/benjaminsteinhart/Desktop/image_seg/validate_masks_morph/"
# MaskDict = {}
# count = 0
# for i in range(1, 61):
#     for file in sorted(glob.glob(path_to_masks + "participant" + str(i) + '/*.jpeg')):
#         dirname, basename = os.path.split(file)
#         image = cv.imread(file)
#         img = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
#         masks = label_extraction(img)
#         # print(basename)
#         tempDict = {basename : masks}
#         MaskDict.update(tempDict)
#         count +=1

# ######


# for i in range(1, 61):
#     for file in sorted(glob.glob(path_to_masks + "participant" + str(i) + '/*.jpeg')):
#         try:
#             # time
#             time_start = time.time()
#             ##
#             dirname, basename = os.path.split(file)
#             mask_img, radius = mask_prediction(file, 40)
#             label_temp = MaskDict[basename[:-4] + "jpeg"]

#             ## Accuracy
#             acc = accuracy_score(label_temp.reshape(-1), mask_img.reshape(-1))
#             print('the accuracy score is', acc)
#             acc_score.append(acc)

#             ## Precision
#             precision = sklearn.metrics.precision_score(label_temp.reshape(-1), mask_img.reshape(-1), labels=(0,255), pos_label=255, average='binary', sample_weight=None, zero_division='warn')
#             print('the precision score is', precision)
#             prec_score.append(precision)

#             ## F1

#             f1 = sklearn.metrics.f1_score(label_temp.reshape(-1), mask_img.reshape(-1), labels=(0,255), pos_label=255, average='binary', sample_weight=None, zero_division='warn')
#             print('the f1 score for is', f1)
#             f1_score_list.append(f1)

#             ## Jaccard

#             jaccard = sklearn.metrics.jaccard_score(label_temp.reshape(-1), mask_img.reshape(-1), labels=(0,255), pos_label=255, average='binary', sample_weight=None, zero_division='warn')
#             print('the jaccard score for threshold is', jaccard)
#             jaccard_list.append(jaccard)

#             ## adding the axis
#             radius_list.append(radius)

#             ## time for completion
#             time_ellapse = time.time() - time_start
#             time_list.append(time_ellapse)

#             ## Adding file name
#             file_names.append(basename)

#             print("mask and labels ready")
#         except Exception:
#             print("exception occured")


# ####


# percentile_list = pd.DataFrame(
#     {'Image':file_names,
#      'Accuracy': acc_score,
#      'Precision': prec_score,
#      'F1': f1_score_list,
#      'Jaccard': jaccard_list,
#      'Radius' : radius_list,
#      'Time' : time_list
#     })


# percentile_list.to_csv(path_or_buf= "/Users/benjaminsteinhart/Desktop/hough_true.csv")

#####################
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import statistics
import os
import glob
import pickle
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
import time
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, accuracy_score, RocCurveDisplay
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import black_tophat, skeletonize, convex_hull_image  # noqa
from skimage.morphology import disk
import scipy
import scipy.ndimage
import random as rng
import math
import sklearn
import tracemalloc # for tracking memory

## Ellipse
time_start = time.time()

############ Image Features #############

# Note; Not being used to train but to check accuracy of images that were not used in training. 
def label_extraction(labeled_img):
    #Now, add a column in the data frame for the Labels
    #For this, we need to import the labeled image
    #But, drop the rows with unlabeled data
    #changing mask to strict binary

    labeled_img[labeled_img < 40] = 0 # from the masks created, pixel values vary slightly but pupil pixels are > 200
    labeled_img[labeled_img > 0] = 255 # max value for gray scale.
    
    #########################################################
    return labeled_img

## Setting path for images

## validation
path_to_masks = "/Users/benjaminsteinhart/Desktop/image_seg/validate_masks_morph/"

## training
# path_to_masks = '/Users/benjaminsteinhart/Desktop/image_seg/manual_seg_clean_all_masks/*jpeg'
### Creating dictionary of true mask labels
MaskDict = {}

count = 0

for i in range(1, 61): # validation
    for file in sorted(glob.glob(path_to_masks + "participant" + str(i) + '/*.jpeg')): # validation
# for file in sorted(glob.glob(path_to_masks)): # training
        dirname, basename = os.path.split(file)
        image = cv.imread(file)
        img = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        masks = label_extraction(img)
        # print(basename)
        tempDict = {basename : masks}
        MaskDict.update(tempDict)
        count +=1

def contour_mask(image):


    #### canny
    max_thresh = 220
    min_thresh = 100
    canny_output = cv.Canny(image, min_thresh, max_thresh, apertureSize = 3)

    contours, hierachy= cv.findContours(canny_output,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    # Find the rotated rectangles and ellipses for each contour
    minRect = [None]*len(contours)
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = cv.fitEllipse(c)

    minor_axis_list = []
    major_axis_list = []
    centerX_list = []
    centerY_list = []
    rotation_list = []
    axis_list_avg = []
    rotation_list_avg = []
    center_list_avg = []
    area_list_avg = []
    area_list = []
    eccentricity_list = []
    eccentricity_list_avg = []
    for i, c in enumerate(contours):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # ellipse
        if c.shape[0] > 10:
            # area of ellipse
            minor_axis = minEllipse[i][1][0]
            major_axis = minEllipse[i][1][1]
            centerX = minEllipse[i][0][0]
            centerY = minEllipse[i][0][1]
            axis = list(minEllipse[i][1])
            center = minEllipse[i][0]
            rotation = minEllipse[i][2]
            center_rounded = [round(number) for number in center]
            area = math.pi * major_axis * minor_axis
            #print(area)
            # e = the eccentricity of the ellipse. e2 = 1 - b2/a2.; will assume pupil object is closest to zero
            e = 1 - ((minor_axis/2)**2)/((major_axis/2)**2)
            #print(e)
            if area > 3000 and e < 0.6:
                #print("the minor axis is " , minor_axis)
                #print("the major axis is ",major_axis)
                minor_axis_list.append(minor_axis)
                major_axis_list.append(major_axis)
                centerX_list.append(centerX)
                centerY_list.append(centerY)
                rotation_list.append(rotation)
                area_list.append(area)
                eccentricity_list.append(e)
        # rotated rectangle
        box = cv.boxPoints(minRect[i])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)

    return minor_axis_list, major_axis_list, centerX_list, centerY_list, rotation_list, area_list, eccentricity_list



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
    color = (0, 150, 255)

    # Line thickness of -1 px
    thickness = 2

    # Using cv2.ellipse() method
    # Draw a ellipse with blue line borders of thickness of -1 px
    image = cv.ellipse(image, center_coordinates, axesLength, angle,
                            startAngle, endAngle, color, thickness)

    # plt.imshow(image)
    # plt.title("mask prediction")
    # plt.show()

    return image

acc_score = []
prec_score = []
f1_score_list = []
jaccard_list = []
axis_list_minor = []
axis_list_major = []
rotation_all = []

time_list = []
count_completed_images = []
file_names = []
e_list = []
a_list = []
completed = 0
count_label = 0


#     ### initialize empty array to store mask predictions ### 
  ### initialize empty array to store mask predictions ### 

for i in range(1, 61): # validation
    for file in sorted(glob.glob(path_to_masks + "participant" + str(i) + '/*.jpeg')): # validation
# for file in sorted(glob.glob(path_to_masks)): # training
    # time
        time_start = time.time()
        ##
        dirname, basename = os.path.split(file)
        minor_axis, major_axis, centerX, centerY, rotation, area, eccentricity =  contour_mask(MaskDict[basename[:-4] + "jpeg"]) # threshold determined from cross fold
        print("length minor axis", len(minor_axis))
        print("minor axis list", minor_axis)
        # for j in range(len(minor_axis)):
        mask_img = mask_prediction(MaskDict[basename[:-4] + "jpeg"], (round(centerX[0]), round(centerY[0])), (round(minor_axis[0]/2), round(major_axis[0]/2)), round(rotation[0]))

        im_overlap = image_overlap(cv.imread(file), (round(centerX[0]), round(centerY[0])), (round(minor_axis[0]/2), round(major_axis[0]/2)), round(rotation[0]))

        label_temp = MaskDict[basename]

        ## Accuracy
        acc = accuracy_score(label_temp.reshape(-1), mask_img.reshape(-1))
        print('the accuracy score is', acc)
        acc_score.append(acc)

        ## Precision
        precision = sklearn.metrics.precision_score(label_temp.reshape(-1), mask_img.reshape(-1), labels=(0,255), pos_label=255, average='binary', sample_weight=None, zero_division='warn')
        print('the precision score is', precision)
        prec_score.append(precision)

        ## F1

        f1 = sklearn.metrics.f1_score(label_temp.reshape(-1), mask_img.reshape(-1), labels=(0,255), pos_label=255, average='binary', sample_weight=None, zero_division='warn')
        print('the f1 score for is', f1)
        f1_score_list.append(f1)

        ## Jaccard

        jaccard = sklearn.metrics.jaccard_score(label_temp.reshape(-1), mask_img.reshape(-1), labels=(0,255), pos_label=255, average='binary', sample_weight=None, zero_division='warn')
        print('the jaccard score for threshold is', jaccard)
        jaccard_list.append(jaccard)
        # if jaccard <0.8:
            ##### visualize ellipse on original image
            # plt.imshow(im_overlap)
            # plt.title(basename)
            # plt.show()
            ####
            # # for exploring bad performers
            # image_original = cv.imread(file)
            # fig, axs = plt.subplots(2, 2)
            # axs[0, 0].imshow(image_original)
            # axs[0, 0].set_title(basename)
            # axs[0, 1].imshow(ImageDict[basename])
            # axs[0, 1].set_title('probability - RF')
            # axs[1, 0].imshow(erosion)
            # axs[1, 0].set_title('Image after Binary Morphology')
            # axs[1, 1].imshow(im_overlap)
            # axs[1, 1].set_title('Predicted Ellipse')
            # plt.show()

        ## adding the axis
        axis_list_minor.append(minor_axis[0])
        axis_list_major.append(major_axis[0])
        ## adding e to e list
        e_list.append(eccentricity[0])

        ## adding area
        a_list.append(area[0])

        ## adding rotation 

        rotation_all.append(rotation[0])

        ## time for completion
        # time_ellapse = time.time() - time_start
        # time_list.append(time_ellapse)

        ## Adding file name
        file_names.append(basename)

        print("mask and labels ready")
        count_label +=1
    

percentile_list = pd.DataFrame(
     {'file': file_names,
     'Accuracy': acc_score,
     'Precision': prec_score,
     'F1 Score': f1_score_list,
     'Jaccard Score': jaccard_list,
     'Minor axis': axis_list_minor,
     'Major axis': axis_list_major,
     'eccentricity':e_list,
     "rotation":rotation_all,
     "ellipse_area":  a_list})


percentile_list.to_csv(path_or_buf= "/Users/benjaminsteinhart/Desktop/image_seg/results/true_output_ellipse_validation.csv")



# # area

# ## Ellipse
# time_start = time.time()

# ############ Image Features #############


# def label_extraction(labeled_img):
#      ######################################
#     #Now, add a column in the data frame for the Labels
#     #changing mask to strict binary
#     labeled_img[labeled_img<50] = 0
#     labeled_img[labeled_img>0] = 1
#     #reshape to one column
#     # labeled_img1 = labeled_img.reshape(-1)
#     #########################################################
#     #Define the dependent variable that needs to be predicted (labels)
#     return labeled_img

# ## Setting path for images
# path_to_masks = "/Users/benjaminsteinhart/Desktop/image_seg/validate_masks_morph/"


# ### Creating dictionary of true mask labels
# MaskDict = {}

# count = 0
# for i in range(1, 61):
#     for file in sorted(glob.glob(path_to_masks + "participant" + str(i) + '/*.jpeg')):
#         dirname, basename = os.path.split(file)
#         image = cv.imread(file)
#         img = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
#         masks = label_extraction(img)
#         # print(basename)
#         tempDict = {basename : masks}
#         MaskDict.update(tempDict)
#         count +=1



# time_list = []
# file_names = []
# a_list = []
# completed = 0
# count_label = 0


# #     ### initialize empty array to store mask predictions ### 
#   ### initialize empty array to store mask predictions ### 
# for i in range(1, 61):
#     try:
#         for file in sorted(glob.glob(path_to_masks + "participant" + str(i) + '/*.jpeg')):
#             dirname, basename = os.path.split(file)
#             area = MaskDict[basename]
#             area = np.sum(area)
#             a_list.append(area)
#             ## Adding file name
#             file_names.append(basename + str(i))
        
#     except Exception:
#         print("exception occured")


# percentile_list = pd.DataFrame(
#      {'file': file_names,
#      'pixel_area': a_list
# })


# percentile_list.to_csv(path_or_buf= "/Users/benjaminsteinhart/Desktop/true_output_pixels.csv")

