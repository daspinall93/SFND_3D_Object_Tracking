#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"

enum class DetectorType
{
    Shitomasi,
    Harris,
    Fast,
    Brisk,
    Orb,
    Akaze,
    Sift
};
const std::unordered_map<DetectorType, std::string> DetectorStrings {
    { DetectorType::Shitomasi, "Shitomasi" },
    { DetectorType::Harris, "Harris" },
    { DetectorType::Fast, "Fast" },
    { DetectorType::Brisk, "Brisk" },
    { DetectorType::Orb, "Orb" },
    { DetectorType::Akaze, "Akaze" },
    { DetectorType::Sift, "Sift" }
};
enum class DescriptorType
{
    Brisk,
    Brief,
    Orb,
    Freak,
    Akaze,
    Sift
};
const std::unordered_map<DescriptorType, std::string> DescriptorStrings {
    { DescriptorType::Brisk, "Brisk" },
    { DescriptorType::Brief, "Brief" },
    { DescriptorType::Orb, "Orb" },
    { DescriptorType::Freak, "Freak" },
    // { DescriptorType::Akaze, "Akaze" },
    { DescriptorType::Sift, "Sift" }
};
enum class MatcherType
{
    BF,
    FLANN
};

enum class DescriptorClass
{
    Binary,
    HOG
};

enum class SelectorType
{
    NN,
    KNN
};

void detKeypointsCv(std::vector<cv::KeyPoint> &keyPoints, cv::Mat &img, const cv::Ptr<cv::FeatureDetector>& featureDetector, bool bVis=false);
void detKeypointsHarris(std::vector<cv::KeyPoint> &keyPoints, cv::Mat &img, bool bVis=false);
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keyPoints, cv::Mat &img, bool bVis=false);
double detKeypointsModern(std::vector<cv::KeyPoint> &keyPoints, cv::Mat &img, DetectorType detectorType, bool bVis=false);
double descKeypoints(std::vector<cv::KeyPoint> &keyPoints, cv::Mat &img, cv::Mat &descriptors, DescriptorType descriptorType);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, DescriptorClass descriptorClass, MatcherType matcherType, SelectorType selectorType);

#endif /* matching2D_hpp */
