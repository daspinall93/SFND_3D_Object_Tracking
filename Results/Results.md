## FP.1 Match 3D Objects
The matchBoundingBoxes function has been implemented to match bounding boxes based on which boxes have the highest umber of keypoint correspondences between one frame and the next. The result of this unction can be seen in the value NumMatchBoxes which is consistantly greater than 6 for each frame.

## FP.2 Compute Lidar-based TTC
The computeTTCLidar function has been implemented to calculate the TTC based purely on lidar data. This is done by using the change in closest lidar point in the x dimension. The lidar distances are filtered using the Median absolute deviation (MADS). The results can be seen in lidarTTC.png which show the ttc decreasing as the frame index increases.

## FP.3 Associate Keypoint Correspondences with Bounding Boxes
The clusterKptMatchesWithROI function has been implemented to associate keypoints with detected bounding boxes. This is done by detecting keypoint matches where both the previous and this frames keypoints are located within the bounding box. The keypoints are then filtered using MADS and finally added to the respective bounding boxes keypoints and matches structures.

## FP.4 Compute Camera-based TTC
The computeTTCCamera function has been implemented to calculate the TTC using the ratio between current and previous keypoint distances. Once again these distance ratios are filtered using MADS to remove outliers. Finally the median distance ratio is used for the TTC calculation. A selection of the best results can be seen in CameraTTC.png. As can be seen the trajectory is similar to that of the lidar TTC (shown in the same figure).

## FP.5 Performance Evaluation 1
Looking at the lidar TTC image it can be seen that there is a significant spike in the ttc at image index 3 and 8. This is due to the way that the lidar TTC is calculated which is based purely on the speed determined by change in position. It can be seen in lidarTTC.png that rather than the change in distance (delta) being a consistent value there is high variance corresponding to the spikes seen in the TTC value. As this is all the TTC is based on, noisy values in this will lead to noisy values in the TTC.

## FP.6 Performance Evaluation 2
The performance of the camera based TTC can be seen in the SortedMeanError.csv file which shows the error characterstics of the different detector and descriptor combinations. The best results are shown in cameraTTC.png. In the SortedMeanError.csv file it can also be seen towards the bottom that some combinations provided very poor results. These are generally when using the Harris and Orb detectors. From the previous mid-term assigment (SFND_2D_Feature_Tracking.csv) it can be seen that these detectors were typically the ones with the least number of keypoints being detected. This therefore introduces greater noise and a greater probability of the keypoint not successfully associated with the bounding box at all. This then leads to the high number of NaN values when using these detectors.