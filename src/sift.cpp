#include "../include/sift.hpp"
#include <spdlog/spdlog.h>
#include <random>
#include <Python.h>
#include <iostream>

SIFT::SIFT(){

}
// Convert image to grayscale
cv::Mat cv2gray(const cv::Mat &img) {
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY); // Convert BGR image to grayscale
    return gray;
}

cv::Mat cv2hsv(const cv::Mat &img) {
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    // Split the HSV image into three channels
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsv, hsvChannels);

    // Save each channel as a separate JPG image
    cv::imwrite("./test/H_channel.jpg", hsvChannels[0]); // Hue channel
    cv::imwrite("./test/S_channel.jpg", hsvChannels[1]); // Saturation channel
    cv::imwrite("./test/V_channel.jpg", hsvChannels[2]); // Value channel

    return hsv;
}

// Detect SIFT keypoints and compute descriptors
void detect_sift(const cv::Mat &channel, std::vector<cv::KeyPoint> &kp, cv::Mat &des) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(); // Create SIFT detector
    sift->detectAndCompute(channel, cv::noArray(), kp, des); // Detect keypoints and compute descriptors
}

// Match descriptors using FLANN-based matcher
std::vector<std::vector<cv::DMatch>> match_sift(const cv::Mat &src_des, const cv::Mat &dst_des) {
    if (src_des.empty()) {
        spdlog::error("First descriptor matrix is empty.");
        return {};
    }
    if (dst_des.empty()) {
        spdlog::error("Second descriptor matrix is empty.");
        return {};
    }
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(src_des, dst_des, knnMatches, 2); // Perform KNN matching with k=2
    return knnMatches;
}

// Apply the ratio test to filter good matches
std::vector<cv::DMatch> ratio_test(const std::vector<std::vector<cv::DMatch>> &knnMatches, float rt) {
    std::vector<cv::DMatch> goodMatches;
    for (const auto &knnMatch : knnMatches) {
        if (knnMatch[0].distance < rt * knnMatch[1].distance) {
            goodMatches.push_back(knnMatch[0]); // Keep only the good matches
        }
    }
    return goodMatches;
}


double SIFT::compute(const cv::Mat &src_img, const cv::Mat &dst_img, float rt) {
    if (src_img.empty()) { spdlog::error("First input image is empty."); return 0; }
    if (dst_img.empty()) { spdlog::error("Second input image is empty."); return 0; }

    std::vector<cv::KeyPoint> src_kp, dst_kp;
    std::vector<cv::Mat> src_des(3), dst_des(3);

    size_t dst_des_total = 0;
    
    cv::Mat src_hsv = cv2hsv(src_img);
    cv::Mat dst_hsv = cv2hsv(dst_img);

    // Split HSV channels
    std::vector<cv::Mat> src_channels, dst_channels;
    cv::split(src_hsv, src_channels);
    cv::split(dst_hsv, dst_channels);

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    const float ratioThresh = 0.75f;
    
    long num_matches = 0;
    long num_features = 0;

    for (int i = 0; i <= 2; ++i) {
        sift->detectAndCompute(src_channels[i], cv::noArray(), keypoints1, descriptors1);
        sift->detectAndCompute(dst_channels[i], cv::noArray(), keypoints2, descriptors2);

        matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2); // Find the 2 nearest neighbors

        for (const auto& knnMatch : knnMatches) {
            num_features++;
            if (knnMatch[0].distance < ratioThresh * knnMatch[1].distance) {
                num_matches++;
            }
        }
    }

    spdlog::error("We got {} / {}", num_matches, num_features);
    double score = num_matches / (double)num_features;
    if (score < 0.1f)
        return 0;
    return score;
}

std::pair<double, double> SIFT::homography(const cv::Mat& img1, const cv::Mat& img2) {
    // Initialize the SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // Detect keypoints and compute descriptors for both images
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    sift->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    // Use BFMatcher with default params (L2 norm for SIFT)
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2); // Find the 2 nearest neighbors

    // Apply the ratio test to find good matches
    const float ratioThresh = 0.75f;
    std::vector<cv::DMatch> goodMatches;
    for (const auto& knnMatch : knnMatches) {
        if (knnMatch[0].distance < ratioThresh * knnMatch[1].distance) {
            goodMatches.push_back(knnMatch[0]);
        }
    }

    // Extract location of good keypoints in both images
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : goodMatches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }

    // Find the homography matrix using RANSAC
    cv::Mat H;
    if (points1.size() >= 4) { // At least 4 points are needed to compute homography
        H = cv::findHomography(points1, points2, cv::RANSAC);
    } else {
        spdlog::error("Not enough points to compute homography, return default values");
        return std::make_pair(0.0, 0.0);
    }

    double x = H.at<double>(0, 2);
    double y = H.at<double>(1, 2);

    double norm = std::sqrt(x * x + y * y);

    if (norm != 0) {
        x /= norm;
        y /= norm;
    }

    return std::make_pair(x, y);
}