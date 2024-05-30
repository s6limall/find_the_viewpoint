//
// Created by ayush on 5/21/24.
//

#ifndef IMAGE_PROCESSING_HPP
#define IMAGE_PROCESSING_HPP

#include <opencv2/opencv.hpp>
#include <vector>

const float RATIO_THRESH = 0.75f; // Define a constant for ratio test threshold

cv::Mat convertToGrayscale(const cv::Mat &image);
void detectAndComputeSIFT(const cv::Mat &gray, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);
std::vector<std::vector<cv::DMatch>> matchSIFTDescriptors(const cv::Mat &descriptors1, const cv::Mat &descriptors2);
std::vector<cv::DMatch> applyRatioTest(const std::vector<std::vector<cv::DMatch>> &knnMatches, float ratioThresh = RATIO_THRESH);
size_t computeSIFTMatches(const cv::Mat &image1, const cv::Mat &image2, float ratioThresh = RATIO_THRESH);
bool compareImages(const cv::Mat &image1, const cv::Mat &image2);

#endif // IMAGE_PROCESSING_HPP