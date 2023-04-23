# pupil_segmentation_and_analysis

This repository contains code that was used to create a segmentation pipeline from video that captures pupil response to augmented tests inside headset goggles. 


* Example Folder

  + The example folder is included here with a video that is capturing pupils during the bright test administered inside a set of goggles. The goggles and test were designed by Ocular Data Systems, Inc. The company provided this exampled video so that we may provide any interested parties with the ability to see how our segmentation pipeline was utilized. It should be noted that this test video had slight variations compared to the videos used in our research. For example, the test is longer and the video captured is of a different size. Therefore, some of the parameters in the example pipeline were modified slightly to capture the eyes in the video but the overall outline and functionality is the same. Below, we provide a brief description of the example folder contents.

  + example_video.mp4: This is the example video provided by the company. 
  + STEP1 (segmentation_analysis/seg_output.py): This python script takes as input the location of a folder that contains one or more videos. Each video is then broken into individual frames and split into two images capturing the right and left eye. The pupil is then segmented from these images using a variety of image segmentation techniques described in the paper. The segmentation pipeline outputs estimates for the pupil size (estimates currently in pixel size and later converted into percent change)
  + STEP2 (cleaning_and_registration/qgam_cleaning.Rmd): This file takes the output from step one and cleans the data using quantile generalized additive models. From the cleaned points, a smoothed trajectory is created. This file outputs a cleaned trajectory and an image capturing the cleaning process titled, "cleaned_pupil_estimates.pdf", found in the results folder.
  + Step3 (cleaning_and_registration/registration_of_trajectory.Rmd): For the data we collected, the start and end time for each subject was slightly different (see limitations section of the paper) and we therefore had to estimate what the beginning and end of the test were for each subject. This file completes that process and outputs a final trajectory in percent change, which is displayed in the "registered_curve.pdf" document in the results section. 
  + Step4 (scalar_features.Rmd): This file takes the registered curve data from step3 and calculates some scalar features that were used in subsequent analysis. 
  
  + A more detailed description of the segmentation, cleaning, and registration process along with the downstream analyses can be found in the paper.
  
* Segmentation

  + seg_output (STEP1).py: This python script takes as input the location of a folder that contains one or more videos. Each video is then broken into individual frames and split into two images capturing the right and left eye. The pupil is then segmented from these images using a variety of image segmentation techniques described in the paper. The segmentation pipeline outputs estimates for the pupil size (estimates currently in pixel size and later converted into percent change)
  + useful_figures.py: This script was used to create useful figures, many of which are displayed in the paper. The figure that captures the overview of the segmentation process, among others, was created in this script. 
  
* Cleaning and Analysis
  + STEP2 (cleaning_curves.Rmd): This file takes the output from step one and cleans the data using quantile generalized additive models. From the cleaned points, a smoothed trajectory is created. This file outputs a cleaned trajectory for each subject.
  + STEP3 (manual_cleaning.Rmd): The above step did not function for all subjects. Some manual cleaning was required. Points that were clearly not part of the pupil trajectory were manually removed after consensus among researchers was reached. The manual cleaning process is described in the paper. 
  + STEP4 (registration.Rmd): For the data we collected, the start and end time for each subject was slightly different (see limitations section of the paper) and we therefore had to estimate what the beginning and end of the test were for each subject. This file completes that process and outputs a final trajectory in percent change. 
  + STEP5 (scalar_features.Rmd): This file takes the registered curve data from step4 and calculates some scalar features that were used in subsequent analysis. 
  + STEP6 (gee_tables.Rmd): This script takes several of the scalar metrics that were calculated in STEP5 and then analyzes the data using generalized estimating equations (GEE). We analyzed the data at the post event as well as looking at the difference (POST - PRE). The script outputs some summary tables as well as some summary plots. 
  + STEP7 (r_plots.Rmd): Similar to the useful_figures file, this script creates a wide variety of plots that were used in the paper as well as a poster presentation. It creates plots for both scalar features as well as descriptive plots of the process as a whole, taking as inputs the raw and cleaned pupil size estimates. 
  
* Figures
  + includes figures that were used in the paper.
  
  
  