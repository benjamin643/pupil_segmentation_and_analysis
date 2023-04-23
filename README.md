# pupil_segmentation_and_analysis

This repository contains code that was used to create a segmentation pipeline from video that captures pupil response to augmented tests inside headset goggles. 


* Example Folder

  + The example folder is included here with a video that is capturing pupils during the bright test administered inside a set of goggles. The goggles and test were designed by Ocular Data Systems, Inc. The company provided this exampled video so that we may provide any interested parties with the ability to see how our segmentation pipeline was utilized. It should be noted that this test video had slight variations compared to the videos used in our research. For example, the test is longer and the video captured is of a different size. Therefore, some of the parameters in the example pipeline were modified slightly to capture the eyes in the video but the overall outline and functionality is the same. Below, we provide a brief description of the example folder contents.

  + example_video.mp4: This is the example video provided by the company. 
  + STEP1 (segmentation_analysis/seg_output.py): This python script takes as input the location of a folder that contains one or more videos. Each video is then broken into individual frames and split into two images capturing the right and left eye. The pupil is then segmented from these images using a variety of image segmentation techniques described in the paper. The segmentation pipeline outputs estimates for the pupil size (estimates currently in pixel size and later converted into percent change)
  + STEP2 (cleaning_and_registration/qgam_cleaning.Rmd): This file takes the output from step one and cleans the data using quantile generalized additive models. From the cleaned points, a smoothed trajectory is created. This file outputs a cleaned trajectory and an image capturing the cleaning process titled, "cleaned_pupil_estimates.pdf", found in the results folder.
  + Step3 (cleaning_and_registration/registration_of_trajectory.Rmd): For the data we collected, the start and end time for each subject was slightly different (see limitations section of the paper) and we therefore had to estimate what the beginning and end of the test were for each subject. This file completes that process and outputs a final trajectory in percent change, which is displayed in the "registered_curve.pdf" document in the results section. 
  + Step4 (scalar_features.Rmd): This file takes the registered curve data from step3 and calculates some scalar features that were used in subsequent analysis. 
  
  A more detailed description of the segmentation, cleaning, and registration process along with the downstream analyses can be found in the paper.
  