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
void detectAndComputeSIFT(const cv::Mat &gray, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {
    if (gray.empty()) {
        spdlog::error("Input grayscale image is empty.");
        return;
    }
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(); // Create SIFT detector
    sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptors); // Detect keypoints and compute descriptors
}

// Match descriptors using FLANN-based matcher
std::vector<std::vector<cv::DMatch>> matchSIFTDescriptors(const cv::Mat &descriptors1, const cv::Mat &descriptors2) {
    if (descriptors1.empty()) {
        spdlog::error("First descriptor matrix is empty.");
        return {};
    }
    if (descriptors2.empty()) {
        spdlog::error("Second descriptor matrix is empty.");
        return {};
    }
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2); // Perform KNN matching with k=2
    return knnMatches;
}

// Apply the ratio test to filter good matches
std::vector<cv::DMatch> applyRatioTest(const std::vector<std::vector<cv::DMatch>> &knnMatches, float ratioThresh) {
    std::vector<cv::DMatch> goodMatches;
    for (const auto &knnMatch : knnMatches) {
        if (knnMatch[0].distance < ratioThresh * knnMatch[1].distance) {
            goodMatches.push_back(knnMatch[0]); // Keep only the good matches
        }
    }
    return goodMatches;
}

// Compute the number of good SIFT matches between two images
size_t computeSIFTMatches(const cv::Mat &image1, const cv::Mat &image2, float ratioThresh) {
    if (image1.empty()) {
        spdlog::error("First input image is empty.");
        return 0;
    }
    if (image2.empty()) {
        spdlog::error("Second input image is empty.");
        return 0;
    }
    cv::Mat gray1 = convertToGrayscale(image1);
    cv::Mat gray2 = convertToGrayscale(image2);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detectAndComputeSIFT(gray1, keypoints1, descriptors1);
    detectAndComputeSIFT(gray2, keypoints2, descriptors2);

    auto knnMatches = matchSIFTDescriptors(descriptors1, descriptors2);
    auto goodMatches = applyRatioTest(knnMatches, ratioThresh);
    return goodMatches.size();
}

// Compare two images for similarity using SIFT feature matching
bool compareImages(const cv::Mat &image1, const cv::Mat &image2) {
    if (image1.empty()) {
        spdlog::error("First input image is empty.");
        return false;
    }
    if (image2.empty()) {
        spdlog::error("Second input image is empty.");
        return false;
    }
    if (image1.size() != image2.size()) {
        spdlog::error("Input images do not match in size.");
        return false;
    }
    if (image1.type() != image2.type()) {
        spdlog::error("Input images do not match in type.");
        return false;
    }

    size_t goodMatches = computeSIFTMatches(image1, image2);
    constexpr size_t minGoodMatches = 10;

    bool result = goodMatches >= minGoodMatches;
    spdlog::info("Comparison result: {} ({} good matches, minimum required: {}).", result, goodMatches, minGoodMatches);
    return result;
}