
#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unordered_set>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

double median(std::vector<double> values)
{
    // Find middle value of the sorted list
    std::sort(values.begin(), values.end());
    long medIndex = floor(values.size() / 2.0);
    return values.size() % 2 == 0 ? (values[medIndex - 1] + values[medIndex]) / 2.0 : values[medIndex];
}

double mads(const std::vector<double>& values, double medianVal)
{
    // Calculate distance from the median and store
    std::vector<double> medAbsDev;
    for(auto value : values)
        medAbsDev.push_back(std::fabs(value - medianVal));
    
    auto medMedAbsDev = median(medAbsDev);
    return 1.4826 * medMedAbsDev;
}

bool isOutlier(double value, double median, double mads)
{
    // Determine if is an outlier based on mads score
    if(std::abs(value - median) / mads < 3 * std::abs(mads))
        return false;

    return true;
}

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Determine keypoints within bounding box
    unordered_set<int> containedKeyPointIdxs;
    for (int i = 0; i < kptMatches.size(); i++)
    {
        // Only consider keypoint where both the previous and current keypoints are contained within box
        // trainIdx = current
        // queryIdx = previous
        const cv::DMatch& match = kptMatches.at(i);
        if (boundingBox.roi.contains(kptsCurr.at(match.trainIdx).pt) && boundingBox.roi.contains(kptsPrev.at(match.queryIdx).pt))  
            containedKeyPointIdxs.emplace(i);
    }
    vector<double> keyPointDistances;
    keyPointDistances.reserve(containedKeyPointIdxs.size());
    for (int i : containedKeyPointIdxs)
        keyPointDistances.push_back(kptMatches.at(i).distance);

    // Remove outlier keypoints
    unordered_set<int> selectedMatches;
    selectedMatches.reserve(containedKeyPointIdxs.size());
    double medianMatchDistance = median(keyPointDistances);
    double madsScore = mads(keyPointDistances, medianMatchDistance);
    for (int i : containedKeyPointIdxs)
    {
        const cv::DMatch& match = kptMatches.at(i);
        if (!isOutlier(match.distance, medianMatchDistance, madsScore))
            selectedMatches.emplace(i);
    }

    // Add matches and points to the bounding box
    boundingBox.keypoints.reserve(selectedMatches.size());
    boundingBox.kptMatches.reserve(selectedMatches.size());
    for (int i : selectedMatches)
    {
        const cv::DMatch& match = kptMatches.at(i);
        boundingBox.kptMatches.push_back(match);
        boundingBox.keypoints.push_back(kptsCurr.at(match.trainIdx));
    }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
double computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, cv::Mat *visImg)
{
    // Loop over keypoints, calculate distance ratios 
    std::vector<double> distRatios;
    distRatios.reserve(kptMatches.size());
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    {
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        {            
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // Calculate distances and their ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            double minDist = 100.0;
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    // Only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
        return numeric_limits<double>::quiet_NaN();

    // Determine outliers
    double medianDistRatio = median(distRatios);
    double distRatioMADSScore = mads(distRatios, medianDistRatio);
    std::vector<double> validDistRatios;
    validDistRatios.reserve(distRatios.size());
    for(auto distRatio : distRatios)
    {
        if(!isOutlier(distRatio, medianDistRatio, distRatioMADSScore))
            validDistRatios.push_back(distRatio);
    }

    if (validDistRatios.size() == 0)
        return numeric_limits<double>::quiet_NaN();

    double medianValidDistRatio = median(validDistRatios);

    double dt = 1 / frameRate;
    return -dt / (1 - medianValidDistRatio);
}

LidarTTCResult computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate)
{
    // Calculate distances (x dimension)
    // std::vector<double> prevDistances(lidarPointsPrev.size());
    // std::transform(lidarPointsPrev.begin(), lidarPointsPrev.end(), prevDistances.begin(), [](const LidarPoint& lidarPoint){ return lidarPoint.x; });
    std::vector<double> prevDistances;
    prevDistances.reserve(lidarPointsPrev.size());
    for (const LidarPoint& point : lidarPointsPrev)
        prevDistances.push_back(point.x);

    // std::vector<double> currDistances(lidarPointsCurr.size());
    // std::transform(lidarPointsCurr.begin(), lidarPointsCurr.end(), prevDistances.begin(), [](const LidarPoint& lidarPoint){ return lidarPoint.x; });
    std::vector<double> currDistances;
    currDistances.reserve(lidarPointsCurr.size());
    for (const LidarPoint& point : lidarPointsCurr)
        currDistances.push_back(point.x);

    // Filter distances
    double prevDistanceMed = median(prevDistances);
    double currDistanceMed = median(currDistances);
    double prevDistanceMADS = mads(prevDistances, prevDistanceMed);
    double currDistanceMADS = mads(currDistances, currDistanceMed); 

    // Use closest estimated distances
    double minPrevDistance = numeric_limits<double>::max();
    for (const LidarPoint& point : lidarPointsPrev)
    {
        if(!isOutlier(point.x, prevDistanceMed, prevDistanceMADS) && minPrevDistance > point.x)
            minPrevDistance = point.x;
    }   

    double minCurrDistance = numeric_limits<double>::max();
    for (const LidarPoint& point : lidarPointsCurr)
    {
        if(!isOutlier(point.x, currDistanceMed, currDistanceMADS) && minCurrDistance > point.x)
            minCurrDistance = point.x;
    }   

    // Calculate TTC
    double dt = 1 / frameRate;

    return {(minCurrDistance * dt) / (minPrevDistance - minCurrDistance), minCurrDistance};
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // Generate  2D array to keep track of which previous and current bounding boxes have highest correspondence
    // int prevCurrBoxIDCount[prevFrame.boundingBoxes.size()][currFrame.boundingBoxes.size()] = {0};
    vector<vector<int>> prevCurrBoxIDCount(prevFrame.boundingBoxes.size(), vector<int>(currFrame.boundingBoxes.size()));
    for (const cv::DMatch& match : matches)
    {
        // Determine which boxes the previous and current key points from the match fall into
        // trainIdx = current
        // queryIdx = previous
        std::vector<int> prevKeyPointBoxIDs, currKeyPointBoxIDs;
        cv::Point2f prevKeyPoint = prevFrame.keypoints.at(match.queryIdx).pt;
        cv::Point2f currKeyPoint = currFrame.keypoints.at(match.trainIdx).pt;
        for (const BoundingBox& box : prevFrame.boundingBoxes)
        {
            if (box.roi.contains(prevKeyPoint))
                prevKeyPointBoxIDs.push_back(box.boxID);
        }
        for (const BoundingBox& box : currFrame.boundingBoxes)
        {
            if (box.roi.contains(currKeyPoint))
                currKeyPointBoxIDs.push_back(box.boxID);
        }

        // Update previous and current box ID count array
        for (auto prevID : prevKeyPointBoxIDs)
        {
            for (auto currID : currKeyPointBoxIDs)
            {
                prevCurrBoxIDCount[prevID][currID]++;
            }
        }
    }

    // Determine box matches
    for (int prevID = 0; prevID < prevFrame.boundingBoxes.size(); prevID++)
    {
        // Select box id with highest number of matching key points
        int maxNumKeyPointMatches = 0;
        int selectedCurrBoxID = 0;
        for (int currID = 0; currID < currFrame.boundingBoxes.size(); currID++)
        {
            if (prevCurrBoxIDCount[prevID][currID] > maxNumKeyPointMatches)
            {
                maxNumKeyPointMatches = prevCurrBoxIDCount[prevID][currID];
                selectedCurrBoxID = currID;
            }
        }
        bbBestMatches[prevID] = selectedCurrBoxID;
    }
}
