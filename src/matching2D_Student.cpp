#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, DescriptorClass descriptorClass, MatcherType matcherType, SelectorType selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    switch (matcherType)
    {
    case MatcherType::BF:
    {
        cv::NormTypes normType = descriptorClass == DescriptorClass::HOG ? cv::NORM_L2 : cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        break;
    }
    
    case MatcherType::FLANN:
    {
        if (descSource.type() != CV_32F)
        { // Bug
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::FlannBasedMatcher::create();
        break;
    }
    }

    switch(selectorType)
    {
        case SelectorType::NN:
        {
            std::vector<cv::DMatch> unfilteredMatches;
            matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1 

            break;
        }

        case SelectorType::KNN:
        {
            vector<vector<cv::DMatch>> knnMatches;
            matcher->knnMatch(descSource, descRef, knnMatches, 2); // finds the 2 best matches

            // Distance ratio test to filter knn matches
            double minDistRatio = 0.8;
            for (const auto& match : knnMatches)
            {
                if (match[0].distance < minDistRatio * match[1].distance)
                    matches.push_back(match[0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, DescriptorType descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    switch(descriptorType)
    {
        case DescriptorType::Brisk:
        {
            int threshold = 30;        // FAST/AGAST detection threshold score.
            int octaves = 3;           // detection octaves (use 0 to do single scale)
            float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

            extractor = cv::BRISK::create(threshold, octaves, patternScale);
            break;
        }

        case DescriptorType::Brief:
        {
            int nBytes = 32;
            bool orientation = false;
            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(nBytes, orientation);
        }
        
        case DescriptorType::Orb:
        {
            int nfeatures = 500;
            float scaleFactor = 1.2f;
            int nlevels = 8;
            int edgeThreshold = 31;
            int firstLevel = 0;
            int WTA_K = 2;
            auto scoreType = cv::ORB::HARRIS_SCORE;
            int patchSize = 31;
            int fastThreshold = 20;

            extractor = cv::ORB::create(
                nfeatures, scaleFactor,
                nlevels, edgeThreshold,
                firstLevel, WTA_K,
                scoreType, patchSize, 
                fastThreshold);
            break;
        }
        case DescriptorType::Freak:
        {
            bool orientationNormalized = true;
            bool scaleNormalized = true;
            float patternScale = 22.0f;
            int nOctaves = 4;

            extractor = cv::xfeatures2d::FREAK::create(
                orientationNormalized,
                scaleNormalized,
                patternScale,
                nOctaves);
            break;

        }

        case DescriptorType::Akaze: // Throwing exception
        {
            auto descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
            int descriptor_size = 0;
            int descriptor_channels = 3;
            float threshold = 0.001f;
            int nOctaves = 4;
            int nOctaveLayers = 4;
            auto diffusivity = cv::KAZE::DIFF_PM_G2;

            extractor = cv::AKAZE::create(
                descriptor_type,
                descriptor_size,
                descriptor_channels,
                threshold,
                nOctaves,
                nOctaveLayers,
                diffusivity);
            break;
        }

        case DescriptorType::Sift:
        {
            int nfeatures = 0;
            int nOctaveLayers = 3;
            double contrastThreshold = 0.04;
            double edgeThreshold = 10;
            double sigma = 1.6;

            extractor = cv::SIFT::create(
                nfeatures,
                nOctaveLayers,
                contrastThreshold,
                edgeThreshold,
                sigma);
            break;
        }
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    double tMs = 1000 * t / 1.0;
    std::cout << static_cast<int>(descriptorType) << "Descriptor extraction in " << tMs << " ms" << endl;
    return tMs;
}

double detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, DetectorType detectorType, bool bVis)
{
    double t = (double)cv::getTickCount();
    string windowName;
    switch(detectorType)
    {
        case DetectorType::Shitomasi:
            detKeypointsShiTomasi(keypoints, img, bVis);
            windowName = "Shi-Tomasi Corner Detector Results";
            break;
        
        case DetectorType::Harris:
            detKeypointsHarris(keypoints, img, bVis);
            windowName = "Harris Corner Detector Results";
            break;

        case DetectorType::Fast:
            detKeypointsCv(keypoints, img, cv::FastFeatureDetector::create(), bVis);
            windowName = "FAST Detector Results";
            break;

        case DetectorType::Brisk:
            detKeypointsCv(keypoints, img, cv::BRISK::create(), bVis);
            windowName = "BRISK Detector Results";
            break;

        case DetectorType::Orb:
            detKeypointsCv(keypoints, img, cv::ORB::create(), bVis);
            windowName = "ORB Detector Results";
            break;

        case DetectorType::Akaze:
            detKeypointsCv(keypoints, img, cv::AKAZE::create(), bVis);
            windowName = "AKAZE Detector Results";
            break;

        case DetectorType::Sift:
            detKeypointsCv(keypoints, img, cv::SIFT::create(), bVis);
            windowName = "SIFT Detector Results";
            break;
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    double tMs = 1000 * t / 1.0;
    cout << "Detection with n=" << keypoints.size() << " keypoints in " << tMs << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    return tMs;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keyPoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Loop through harris detector, check value is local maxima and assign
    float maxOverlap = 0.0;
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            {
                
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                bool overlap = false;
                for (auto & keyPoint : keyPoints)
                {
                    overlap = cv::KeyPoint::overlap(newKeyPoint, keyPoint) > maxOverlap;
                    if (overlap)
                    {                        
                        if (response > keyPoint.response)
                            keyPoint = newKeyPoint;
                        break;
                    }
                }    

                if (!overlap)
                    keyPoints.push_back(std::move(newKeyPoint));            
            }
        }
    }
}

void detKeypointsCv(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, const cv::Ptr<cv::FeatureDetector>& featureDetector, bool bVis)
{
    featureDetector->detect(img, keypoints);
}