//
// Created by ayush on 5/21/24.
//

#ifndef IMAGE_PROCESSING_HPP
#define IMAGE_PROCESSING_HPP

#include <opencv2/opencv.hpp>
#include <vector>

const float RATIO_THRESH = 0.75f; // Define a constant for ratio test threshold

cv::Mat convertToGrayscale(const cv::Mat &image);
void detectAndComputeSIFT(const cv::Mat &gray, std::vector<cv::KeyPoint> &keypoints, cv::Mat &des);
std::vector<std::vector<cv::DMatch>> matchSIFTDescriptors(const cv::Mat &des1, const cv::Mat &des2);
std::vector<cv::DMatch> applyRatioTest(const std::vector<std::vector<cv::DMatch>> &knnMatches, float rt = RATIO_THRESH);
size_t computeSIFTMatches(const cv::Mat &src_img, const cv::Mat &dst_img, float rt = RATIO_THRESH);
bool compareImages(const cv::Mat &src_img, const cv::Mat &dst_img);
double computeSIFTMatchRatio(const cv::Mat &src_img, const cv::Mat &dst_img, float rt = RATIO_THRESH);
cv::Mat calculateTransformation(const cv::Mat &src_img, const cv::Mat &dst_img, float rt = RATIO_THRESH);

#endif // IMAGE_PROCESSING_HPP