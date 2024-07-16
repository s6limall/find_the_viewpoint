//
// Created by ayush on 5/21/24.
//

#include "../include/image.hpp"
#include <spdlog/spdlog.h>

// Convert image to grayscale
cv::Mat convertToGrayscale(const cv::Mat &image) {
    if (image.empty()) {
        spdlog::error("Input image is empty.");
        return cv::Mat();
    }
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY); // Convert BGR image to grayscale
    return gray;
}

// Detect SIFT keypoints and compute descriptors
void detectAndComputeSIFT(const cv::Mat &gray, std::vector<cv::KeyPoint> &kp, cv::Mat &des) {
    if (gray.empty()) {
        spdlog::error("Input grayscale image is empty.");
        return;
    }
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(); // Create SIFT detector
    sift->detectAndCompute(gray, cv::noArray(), kp, des); // Detect keypoints and compute descriptors
}

// Match descriptors using FLANN-based matcher
std::vector<std::vector<cv::DMatch>> matchSIFTDescriptors(const cv::Mat &des1, const cv::Mat &des2) {
    if (des1.empty()) {
        spdlog::error("First descriptor matrix is empty.");
        return {};
    }
    if (des2.empty()) {
        spdlog::error("Second descriptor matrix is empty.");
        return {};
    }
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(des1, des2, knnMatches, 2); // Perform KNN matching with k=2
    return knnMatches;
}

// Apply the ratio test to filter good matches
std::vector<cv::DMatch> applyRatioTest(const std::vector<std::vector<cv::DMatch>> &knnMatches, float rt) {
    std::vector<cv::DMatch> goodMatches;
    for (const auto &knnMatch : knnMatches) {
        if (knnMatch[0].distance < rt * knnMatch[1].distance) {
            goodMatches.push_back(knnMatch[0]); // Keep only the good matches
        }
    }
    return goodMatches;
}

// Compute the number of good SIFT matches between two images
size_t computeSIFTMatches(const cv::Mat &src_img, const cv::Mat &dst_img, float rt) {
    if (src_img.empty()) {
        spdlog::error("First input image is empty.");
        return 0;
    }
    if (dst_img.empty()) {
        spdlog::error("Second input image is empty.");
        return 0;
    }
    cv::Mat gray1 = convertToGrayscale(src_img);
    cv::Mat gray2 = convertToGrayscale(dst_img);

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    detectAndComputeSIFT(gray1, kp1, des1);
    detectAndComputeSIFT(gray2, kp2, des2);

    auto knnMatches = matchSIFTDescriptors(des1, des2);
    auto goodMatches = applyRatioTest(knnMatches, rt);
    return goodMatches.size();
}

double computeSIFTMatchRatio(const cv::Mat &src_img, const cv::Mat &dst_img, float rt) {
    if (src_img.empty()) {
        spdlog::error("Candiate image is empty.");
        return 0;
    }
    if (dst_img.empty()) {
        spdlog::error("Target image is empty.");
        return 0;
    }
    cv::Mat gray1 = convertToGrayscale(src_img);
    cv::Mat gray2 = convertToGrayscale(dst_img);

    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    detectAndComputeSIFT(gray1, kp1, des1);
    detectAndComputeSIFT(gray2, kp2, des2);

    auto knnMatches = matchSIFTDescriptors(des1, des2);
    auto goodMatches = applyRatioTest(knnMatches, rt);

    double matchRatio = kp1.empty() ? 0.0 : static_cast<double>(goodMatches.size()) / kp1.size();
    spdlog::info("Ratio: {}, {}/{} with {}",matchRatio, goodMatches.size(),kp1.size(), rt);

    return matchRatio;
}

// Compare two images for similarity using SIFT feature matching
bool compareImages(const cv::Mat &src_img, const cv::Mat &dst_img) {
    if (src_img.empty()) {
        spdlog::error("First input image is empty.");
        return false;
    }
    if (dst_img.empty()) {
        spdlog::error("Second input image is empty.");
        return false;
    }
    if (src_img.size() != dst_img.size()) {
        spdlog::error("Input images do not match in size.");
        return false;
    }
    if (src_img.type() != dst_img.type()) {
        spdlog::error("Input images do not match in type.");
        return false;
    }

    size_t goodMatches = computeSIFTMatches(src_img, dst_img);
    constexpr size_t minGoodMatches = 10;

    bool result = goodMatches >= minGoodMatches;
    spdlog::info("Comparison result: {} ({} good matches, minimum required: {}).", result, goodMatches, minGoodMatches);
    return result;
}

cv::Mat calculateTransformation(const cv::Mat &src_img, const cv::Mat &dst_img, float rt) {
    // Convert images to grayscale
    cv::Mat srcGray = convertToGrayscale(src_img);
    cv::Mat dstGray = convertToGrayscale(dst_img);

    // Detect SIFT keypoints and compute descriptors
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    detectAndComputeSIFT(srcGray, kp1, des1);
    detectAndComputeSIFT(dstGray, kp2, des2);

    // Match descriptors using FLANN-based matcher
    auto knnMatches = matchSIFTDescriptors(des1, des2);

    // Apply the ratio test to filter good matches
    auto goodMatches = applyRatioTest(knnMatches, rt);

    // Extract the matched keypoints
    std::vector<cv::Point2f> points1, points2;
    for (const auto &match : goodMatches) {
        points1.push_back(kp1[match.queryIdx].pt);
        points2.push_back(kp2[match.trainIdx].pt);
    }

    // Calculate the homography matrix using RANSAC
    cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC);

    if (H.empty()) {
        spdlog::error("Homography calculation failed.");
        return cv::Mat();
    }

    return H;
}