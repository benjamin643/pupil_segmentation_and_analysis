import cv2 as cv
import numpy as np
from skimage.feature import canny, peak_local_max
from skimage import data, color
import glob
import os
import pandas as pd
import time
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, disk
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import math
from skimage.filters import threshold_multiotsu

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



### starts recording time for entire program to run
my_time = time.time()
##### This is a function that implements several image processing techniques to create 
##### a binary mask from an input image and then fits an ellipse around the binary image
##### to estimate pupil size. Takes as input gamma value for power-law transform,
##### image morphology disk size, median filter kernel size,
##### lower and upper quantile to truncate histogram to create more balanced classes for otsu
##### thresholding, and an index (0) that indicates grabbing the lowest class as pupil 
##### estimate. Function returns elllipse info capturing pupil size/location. 
##### returns: major, minor axis size. eccentricity. rotation. location

def contour_mask(image, gamma_value, morph_disk, median_thresh, quant1, quant2, otsu_index):

    # image = cv.imread(file)
    # img = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    img_med = ndi.median_filter(image, size=median_thresh)

    log_image = np.array((255/255**gamma_value)*(img_med**(gamma_value)),dtype='uint8')


    thresh_mask = np.logical_and(log_image > 5, log_image < 255)

    ots1 = np.quantile(log_image[thresh_mask].ravel(), quant1)
    ots2 = np.quantile(log_image[thresh_mask].ravel(), quant2)


    # ## Otsu multiple 
    mask = np.logical_and(log_image > ots1, log_image < ots2)
    # thresh = threshold_otsu(log_image[mask])
    # print('threshold is',thresh)

    ## Multi level Otsu
    thresholds = threshold_multiotsu(log_image[mask])
    # print("thresholds", thresholds)


    img_bin = np.zeros((log_image.shape))
    img_bin[log_image <= thresholds[0]] = 255
    img_bin = img_bin.astype('int64')



    footprint = disk(morph_disk)

    closed = closing(img_bin, footprint)

    opened = opening(closed, footprint)
    opened = np.uint8(opened)


    minor_axis_list = []
    major_axis_list = []
    centerX_list = []
    centerY_list = []
    rotation_list = []
    area_list = []
    eccentricity_list = []


    canny_output = cv.Canny(opened, 100, 200, apertureSize = 3)
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
            rotation = minEllipse[i][2]
            area = math.pi * major_axis * minor_axis
            #print(area)
            # e = the eccentricity of the ellipse. e2 = 1 - b2/a2.; will assume pupil object is closest to zero
            e = 1 - ((minor_axis/2)**2)/((major_axis/2)**2)
            #print(e)
            if area > 5000/5 and e < 0.5:
                if area <100000/3:

                    #print("the minor axis is " , minor_axis)
                    #print("the major axis is ",major_axis)
                    minor_axis_list.append(minor_axis)
                    major_axis_list.append(major_axis)
                    centerX_list.append(centerX)
                    centerY_list.append(centerY)
                    rotation_list.append(rotation)
                    area_list.append(area)
                    eccentricity_list.append(e)
                    index_delete = []
    for i in range(len(centerX_list)):
        if centerX_list[i] < 40/3 or centerX_list[i] > 410/3:
            index_delete.append(i)
        if centerY_list[i] < 40/3 or centerY_list[i] > 360/3:
            index_delete.append(i)

    index_delete = np.unique(np.asarray(index_delete))
    # print("center x list before",centerX_list )
    for delete in index_delete:
        del centerX_list[delete]
        del centerY_list[delete]
    distance_min = 1000/3
    index_fin = 0
    for i in range(len(centerX_list)):
        dist_center = np.sqrt((centerX_list[i] - 224)**2 + (centerY_list[i] - 199)**2)
        if dist_center < distance_min:
            index_fin = i
            distance_min = dist_center

    return centerX_list[index_fin], centerY_list[index_fin], minor_axis_list[index_fin], major_axis_list[index_fin], rotation_list[index_fin], eccentricity_list[index_fin]

##### This function does the same as contour_mask function above, only now, there is an added step 
##### of seperating the binary image using the watershed algorithm. Additional input
##### is the size of the watershed basin. Returns ellipse info of best estimate for pupil size. 
def contour_mask_water(image, gamma_value, morph_disk, water_thresh, median_thresh, quant1, quant2, otsu_index):

    # image = cv.imread(file)
    # img = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    img_med = ndi.median_filter(image, size=median_thresh)

    log_image = np.array((255/255**gamma_value)*(img_med**(gamma_value)),dtype='uint8')


    thresh_mask = np.logical_and(log_image > 5, log_image < 255)

    ots1 = np.quantile(log_image[thresh_mask].ravel(), quant1)
    ots2 = np.quantile(log_image[thresh_mask].ravel(), quant2)


    # ## Otsu multiple 
    mask = np.logical_and(log_image > ots1, log_image < ots2)
    # thresh = threshold_otsu(log_image[mask])
    # print('threshold is',thresh)

    ## Multi level Otsu
    thresholds = threshold_multiotsu(log_image[mask])
    # print("thresholds", thresholds)


    img_bin = np.zeros((log_image.shape))
    img_bin[log_image <= thresholds[otsu_index]] = 255
    img_bin = img_bin.astype('int64')



    footprint = disk(morph_disk)

    closed = closing(img_bin, footprint)

    opened = opening(closed, footprint)

    distance = ndi.distance_transform_edt(opened)
    coords = peak_local_max(distance, footprint=np.ones((water_thresh, water_thresh)), labels=opened)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=opened)

    temp_labels = np.unique(labels)
    minor_axis_list = []
    major_axis_list = []
    centerX_list = []
    centerY_list = []
    rotation_list = []
    area_list = []
    eccentricity_list = []


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
                if area > 5000/5 and e < 0.5:
                    if area <100000/3:

                        #print("the minor axis is " , minor_axis)
                        #print("the major axis is ",major_axis)
                        minor_axis_list.append(minor_axis)
                        major_axis_list.append(major_axis)
                        centerX_list.append(centerX)
                        centerY_list.append(centerY)
                        rotation_list.append(rotation)
                        area_list.append(area)
                        eccentricity_list.append(e)
                        index_delete = []
    for i in range(len(centerX_list)):
        if centerX_list[i] < 40/3 or centerX_list[i] > 410/3:
            index_delete.append(i)
        if centerY_list[i] < 40/3 or centerY_list[i] > 360/3:
            index_delete.append(i)

    index_delete = np.unique(np.asarray(index_delete))
    # print("center x list before",centerX_list )
    for delete in index_delete:
        del centerX_list[delete]
        del centerY_list[delete]
    distance_min = 1000/3
    index_fin = 0
    for i in range(len(centerX_list)):
        dist_center = np.sqrt((centerX_list[i] - 224/3)**2 + (centerY_list[i] - 199/3)**2)
        if dist_center < distance_min:
            index_fin = i
            distance_min = dist_center

    return centerX_list[index_fin], centerY_list[index_fin], minor_axis_list[index_fin], major_axis_list[index_fin], rotation_list[index_fin], eccentricity_list[index_fin]

## prediction for mask; creates binary image.
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

time_list = []
file_names = []
all_labels = []
center_list_allx = []
center_list_ally = []
minor_list_all = []
major_list_all = []
eccentricity_list_all = []

time_listR = []
center_list_allxR = []
center_list_allyR = []
minor_list_allR = []
major_list_allR = []
eccentricity_list_allR = []
frame_list = []
file_namesR = []


#### path_to_videos defines where your videos / data are. 
path_to_videos = '/Users/benjaminsteinhart/Desktop/example_video/*.mp4'

## Path to output; where you will save csv info. 

path_to_output = "/Users/benjaminsteinhart/Desktop/example_video/data_output/*.csv"


already_output = []
for file in sorted(glob.glob(path_to_output)):
    dirname, basename = os.path.split(file)
    my_file = basename[:-8]
    already_output.append(my_file)
print(already_output)

## creating random list to select participants from 


already_output = []
for file in sorted(glob.glob(path_to_output)):
    dirname, basename = os.path.split(file)
    my_file = basename[:-8]
    already_output.append(my_file)
print(already_output)

## creating random list to select participants from 


count = 0
count_vid = 1
for file in sorted(glob.glob(path_to_videos)):
    start_time_paper = time.time()
    dirname, basename = os.path.split(file)
    name = str(basename[0:7])
    match_time = file[-8:-4]

    video_time = file[-8:]
    time_list = []
    file_names = []
    all_labels = []
    center_list_allx = []
    center_list_ally = []
    minor_list_all = []
    major_list_all = []
    eccentricity_list_all = []

    time_listR = []
    center_list_allxR = []
    center_list_allyR = []
    minor_list_allR = []
    major_list_allR = []
    eccentricity_list_allR = []
    frame_list = []
    file_namesR = []
    print(name+match_time)
    if video_time != "pre1.mp4":
        if (name+match_time) not in  already_output:
            cap= cv.VideoCapture(file)
            i=0
            while(cap.isOpened()):
                ret, frame = cap.read() #If frame is read correctly, it will be True. So you can check end of the video by checking this return value.
                if ret == False:
                    break
                try:
                    time_start = time.time()
                    left_eye = frame[80:214, 83:233] #for right eye, try right bound at 1600

                    left_eye = cv.cvtColor(left_eye,cv.COLOR_BGR2GRAY) # converts color to grayscale
                    print("eyes segmented")
                    centerX, centerY, minor_axis, major_axis, rotation, eccentricity =  contour_mask(left_eye, .7, 8, 15, 0.01, 0.2, 0) # threshold determined from training (96)
                    # print("centerX is ", centerX)
                    # print("major axis is", major_axis)
                    mask_img = mask_prediction(left_eye, (round(centerX), round(centerY)), (round(minor_axis/2), round(major_axis/2)), round(rotation))
                    ## adding the axi
                    minor_list_all.append(minor_axis)
                    major_list_all.append(major_axis)
                    print(major_list_all)
                    # time for completion
                    time_fin_left =  time.time() - time_start
                    time_list.append(time_fin_left)
                    ## Adding file name
                    file_names.append(basename[0:7] + basename[19:23] + "_pupil_left" + str(i))
                    # print(basename[0:7] + basename[19:23] + "_pupil_left" + str(i))
                    # # ## adding center
                    center_list_allx.append(centerX)
                    center_list_ally.append(centerY)
                    ## adding eccentricity
                    eccentricity_list_all.append(eccentricity)

                except: ### if the above method did not work, a common issue was that the estimate was too large because the pupil connects to the iris. We implmement the watershed algorithm to try to seperate them
                    print("exception")
                    try:
                        time_start = time.time()
                        left_eye = frame[80:214, 83:233] ## captures left eye
                        left_eye = cv.cvtColor(left_eye,cv.COLOR_BGR2GRAY) # converts color to grayscale

                        centerX, centerY, minor_axis, major_axis, rotation, eccentricity =  contour_mask_water(left_eye, .7, 8, 25, 15, 0.01, 0.2, 0) # same function as above but now implements watershed algorithm
                        ## adding the axis                            
                        ##### add values from estimates to the storage vectors for a given subject ############ 
                        minor_list_all.append(minor_axis)
                        major_list_all.append(major_axis)
                        # time for completion
                        time_fin_left =  time.time() - time_start
                        time_list.append(time_fin_left)
                        ## Adding file name
                        file_names.append(basename[0:7] + basename[19:23] + "_pupil_left" + str(i))
                        # print(basename[0:7] + basename[19:23] + "_pupil_left" + str(i))
                        # # ## adding center
                        center_list_allx.append(centerX)
                        center_list_ally.append(centerY)
                        ## adding eccentricity
                        eccentricity_list_all.append(eccentricity)
                        mask_img = mask_prediction(left_eye, (round(centerX), round(centerY)), (round(minor_axis/2), round(major_axis/2)), round(rotation))


                    except:
                        try:
                            time_start = time.time()
                            # print("eyes about to be segmented")
                            left_eye = frame[80:214, 83:233] # captures left eye

                            left_eye = cv.cvtColor(left_eye,cv.COLOR_BGR2GRAY) # converts color to grayscale
      
                            centerX, centerY, minor_axis, major_axis, rotation, eccentricity =  contour_mask_water(left_eye, .7, 8, 25, 15, 0.01, 0.2, 1) # final attempt to estimate ellipse and uses higher otsu threshold to create binary image
                            ## adding the axis                     
                            ##### add values from estimates to the storage vectors for a given subject ############ 
                            minor_list_all.append(minor_axis)
                            major_list_all.append(major_axis)
                            # time for completion
                            time_fin_left =  time.time() - time_start
                            time_list.append(time_fin_left)
                            ## Adding file name
                            file_names.append(basename[0:7] + basename[19:23] + "_pupil_left" + str(i))
                            # # ## adding center
                            center_list_allx.append(centerX)
                            center_list_ally.append(centerY)
                            ## adding eccentricity
                            eccentricity_list_all.append(eccentricity)
                            mask_img = mask_prediction(left_eye, (round(centerX), round(centerY)), (round(minor_axis/2), round(major_axis/2)), round(rotation))

                        except:

                            # print("exception")
                            minor_list_all.append("NA")
                            major_list_all.append("NA")
                            # time for completion
                            time_fin_left =  time.time() - time_start
                            time_list.append(time_fin_left)
                            ## Adding file name
                            file_names.append(basename[0:7] + basename[19:23] + "_pupil_left" + str(i))
                            # print(basename[0:7] + basename[19:23] + "_pupil_left" + str(i))
                            # # ## adding center
                            center_list_allx.append("NA")
                            center_list_ally.append("NA")
                            ## adding eccentricity
                            eccentricity_list_all.append("NA")
                try:
                    time_start = time.time()

                    #### right eye
                    right_eye = frame[80:214, 400:550] # captures right eye

                    right_eye = cv.cvtColor(right_eye,cv.COLOR_BGR2GRAY) # converts color to grayscale
                    centerXR, centerYR, minor_axisR, major_axisR, rotationR, eccentricityR =  contour_mask(right_eye, .7, 8, 15, 0.01, 0.2, 0) # threshold determined from training (96)
                    # print('Ellipses fit')

                    time_fin_right = time.time() - time_start
                     ##### add values from estimates to the storage vectors for a given subject ############ 
                    ## adding the axis
                    minor_list_allR.append(minor_axisR)
                    major_list_allR.append(major_axisR)
                    # time for completion
                    time_fin_right = time.time() - time_start
                    time_listR.append(time_fin_right)
                    ## Adding file name
                    file_namesR.append(basename[0:7] + basename[19:23] + "_pupil_right" + str(i))
                    # ## adding center
                    center_list_allxR.append(centerXR)
                    center_list_allyR.append(centerYR)
                    ## adding eccentricity
                    eccentricity_list_allR.append(eccentricityR)

                except:
                    try:
                        time_start = time.time()

                        #### right eye
                        right_eye = frame[80:214, 400:550] #captures right eye

                        right_eye = cv.cvtColor(right_eye,cv.COLOR_BGR2GRAY) # converts color to grayscale
                        centerXR, centerYR, minor_axisR, major_axisR, rotationR, eccentricityR =  contour_mask_water(right_eye, .7, 8, 25, 15, 0.01, 0.2, 0) # threshold determined from training (96)

                        time_fin_right = time.time() - time_start
                        ##### add values from estimates to the storage vectors for a given subject ############ 

                        ## adding the axis
                        minor_list_allR.append(minor_axisR)
                        major_list_allR.append(major_axisR)
                        # time for completion
                        time_fin_right = time.time() - time_start
                        time_listR.append(time_fin_right)
                        ## Adding file name
                        file_namesR.append(basename[0:7] + basename[19:23] + "_pupil_right" + str(i))
                        # ## adding center
                        center_list_allxR.append(centerXR)
                        center_list_allyR.append(centerYR)
                        ## adding eccentricity
                        eccentricity_list_allR.append(eccentricityR)
                    
                    except:
                        try:
                            time_start = time.time()

                            #### right eye
                            right_eye = frame[80:214, 400:550] # captures right eye

                            right_eye = cv.cvtColor(right_eye,cv.COLOR_BGR2GRAY) # converts color to grayscale
                            centerXR, centerYR, minor_axisR, major_axisR, rotationR, eccentricityR =  contour_mask_water(right_eye, .7, 8, 25, 15, 0.01, 0.2, 1) # threshold determined from training (96)

                            time_fin_right = time.time() - time_start
                            ##### add values from estimates to the storage vectors for a given subject ############ 

                            ## adding the axis
                            minor_list_allR.append(minor_axisR)
                            major_list_allR.append(major_axisR)
                            # time for completion
                            time_fin_right = time.time() - time_start
                            time_listR.append(time_fin_right)
                            ## Adding file name
                            file_namesR.append(basename[0:7] + basename[19:23] + "_pupil_right" + str(i))
                            # ## adding center
                            center_list_allxR.append(centerXR)
                            center_list_allyR.append(centerYR)
                            ## adding eccentricity
                            eccentricity_list_allR.append(eccentricityR)
                        except:
                            ## adding the axis
                            minor_list_allR.append("NA")
                            major_list_allR.append("NA")
                            # time for completion
                            time_fin_right = time.time() - time_start
                            time_listR.append(time_fin_right)
                            ## Adding file name
                            file_namesR.append(basename[0:7] + basename[19:23] + "_pupil_right" + str(i))
                            # ## adding center
                            center_list_allxR.append("NA")
                            center_list_allyR.append("NA")
                            ## adding eccentricity
                            eccentricity_list_allR.append("NA")

                i += 1
                print(i)

            cap.release()
            cv.destroyAllWindows()
            count+=1
            ##### adds each subject specific video results to a data frame to be saved that will have all estimates for each frame for each subject
            percentile_list = pd.DataFrame(
            {'Image':file_names,
            'Time' : time_list,
            'centerx': center_list_allx,
            'centery': center_list_ally,
            'minor_axis': minor_list_all,
            'major_axis': major_list_all,
            "eccentricity": eccentricity_list_all,
            'ImageR':file_namesR,
            'TimeR' : time_listR,
            'centerxR': center_list_allxR,
            'centeryR': center_list_allyR,
            'minor_axisR': minor_list_allR,
            'major_axisR': major_list_allR,
            "eccentricityR": eccentricity_list_allR
            })
            final_time = time.time() - start_time_paper
            print("time to finish one participant video",final_time)
            name = str(basename[0:7])
            percentile_list.to_csv(path_or_buf= "/Users/benjaminsteinhart/Desktop/example_video/data_output/pupil_predictions.csv")
    count_vid +=1 
