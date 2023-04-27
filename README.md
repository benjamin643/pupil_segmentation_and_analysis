# pupil_segmentation_and_analysis

This repository contains code was used to reproduce results from the paper, ``[A Video Segmentation Pipeline for Assessing Changes in Pupil
Response to Light After Cannabis Consumption](https://www.biorxiv.org/content/biorxiv/early/2023/03/21/2023.03.17.533144.full.pdf)'',  which describes a pipeline to analyze videos that capture pupil response to light inside headset goggles. 


## Example Video Folder

The example folder is included here with a video that is capturing pupils during the bright test administered inside a set of goggles. The goggles and test were designed by Ocular Data Systems, Inc. This example video is provided so that any interested parties can reproduce the segmentation portion of our video analysis pipeline. It should be noted that this test video had slight variations compared to the videos used in our research. For example, the test is longer and the video captured is of a different size. Therefore, some of the parameters in the example pipeline were modified slightly to capture the eyes in the video but the overall outline and functionality is the same. Below, we provide a brief description of the example folder contents.

* example_video.mp4: This is the example video provided by the company. 
* STEP1 (segmentation_analysis/seg_output.py): This python script takes as input the location of a folder that contains one or more videos. Each video is then broken into individual frames and split into two images capturing the right and left eye. The pupil is then segmented from these images using a variety of image segmentation techniques described in the paper. The segmentation pipeline outputs estimates for the pupil size (estimates currently in pixel size and later converted into percent change)
* STEP2 (cleaning_and_registration/qgam_cleaning.Rmd): This file takes the output from step one and cleans the data using quantile generalized additive models. From the cleaned points, a smoothed trajectory is created. This file outputs a cleaned trajectory and an image capturing the cleaning process titled, "cleaned_pupil_estimates.pdf", found in the results folder.
* Step3 (cleaning_and_registration/registration_of_trajectory.Rmd): For the data we collected, the start and end time for each subject was slightly different (see limitations section of the paper) and we therefore had to estimate what the beginning and end of the test were for each subject. This file completes that process and outputs a final trajectory in percent change, which is displayed in the "registered_curve.pdf" document in the results section. 
* Step4 (scalar_features.Rmd): This file takes the registered curve data from step3 and calculates some scalar features that were used in subsequent analysis. 
  
* A more detailed description of the segmentation, cleaning, and registration process along with the downstream analyses can be found in the paper.
  
##  Segmentation Folder

* STEP1 (seg_output.py): This python script takes as input the location of a folder that contains one or more videos. Each video is then broken into individual frames and split into two images capturing the right and left eye. The pupil is then segmented from these images using a variety of image segmentation techniques described in the paper. The segmentation pipeline outputs estimates for the pupil size (estimates currently in pixel size and later converted into percent change)
* useful_figures.py: This script was used to create useful figures, many of which are displayed in the paper. The figure that captures the overview of the segmentation process, among others, was created in this script. 

## Data Folder

The videos from our paper cannot be made publicly available due to participant privacy concerns. However, in order to make our pipeline as reproducible as possible we have included a dataset of raw pupil size trajectories output from running the videos from our analysis through our segmentation pipeline. This dataset, `raw_pupil_size_predictions.rds`, can be used to reproduce all downstream analyses from our paper.

* `raw_pupil_size_predictions.rds`: pupil size trajectories estimated from video segmentation, before any data cleaning steps
* `scalar_metrics_pupils.rds`: dataset of scalar features extracted from pupil size trajectories
  
## Cleaning and Analysis Folder

* STEP2 (cleaning_curves.Rmd): This file takes the output from step one and cleans the data using quantile generalized additive models. The qgam is used to remove outliers and output a smoothed pupil size trajectory for each video. 
* STEP3 (manual_cleaning.Rmd): The above step did not function for all subjects. Some manual cleaning was required. Points that were clearly not part of the pupil trajectory were manually removed after consensus among researchers was reached. The manual cleaning process is described in the paper. 
* STEP4 (registration.Rmd): For the data we collected, the start and end time for each video was slightly different (see limitations section of the paper) and we therefore had to estimate what the beginning and end of the test were for each video. This file completes that process and outputs a final trajectory in percent change. 
* STEP5 (scalar_features.Rmd): This file takes the registered curve data from step4 and calculates scalar features that summarize each pupil size trajectory and were used in subsequent analysis. 
* STEP6 (gee_tables.Rmd): This script takes several of the scalar metrics that were calculated in STEP5 and then analyzes the data using generalized estimating equations (GEEs). We analyzed the data at the post event as well as looking at the difference (POST - PRE). The script outputs summary tables and summary plots. 
* STEP7 (r_plots.Rmd): Similar to the useful_figures file, this script creates a variety of plots that were used in the paper as well as a poster presentation. It creates plots for both scalar features as well as descriptive plots of the process as a whole, taking as inputs the raw and cleaned pupil size estimates. 
  
## Figures Folder
  
* includes figures that were used in the paper.
  
  
  