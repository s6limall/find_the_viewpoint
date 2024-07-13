// File: processing/image/comparison/feature_comparator.cpp


#include "processing/image/comparison/feature_comparator.hpp"
#include <common/logging/logger.hpp>
#include "config/configuration.hpp"
#include <stdexcept>
#include <future>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace processing::image {

    /*double FeatureComparator::compare(const cv::Mat &image1, const cv::Mat &image2) const {
        try {
            // Check if images are empty
            if (image1.empty() || image2.empty()) {
                LOG_ERROR("One or both images are empty");
                throw std::invalid_argument("One or both images are empty");
            }

            // Log configuration parameters
            const auto extractorType = config::get("feature_detector.type", "SIFT");
            const auto matcherType = config::get("feature_matcher.type", "FLANN");
            LOG_TRACE("Feature Comparator Configuration - Extractor Type: {}, Matcher Type: {}", extractorType,
                      matcherType);

            // Map to store factory functions for extractors
            using ExtractorFactory = std::function<std::unique_ptr<FeatureExtractor>()>;
            static const std::map<std::string, ExtractorFactory> extractorFactoryMap = {
                    {"SIFT", []() { return FeatureExtractor::create<SIFTExtractor>(); }},
                    {"ORB", []() { return FeatureExtractor::create<ORBExtractor>(); }}
            };

            // Map to store factory functions for matchers
            using MatcherFactory = std::function<std::unique_ptr<FeatureMatcher>()>;
            static const std::map<std::string, MatcherFactory> matcherFactoryMap = {
                    {"FLANN", []() { return FeatureMatcher::create<FLANNMatcher>(); }},
                    {"BF", []() { return FeatureMatcher::create<BFMatcher>(); }}
            };

            // Create extractor
            auto extractorIt = extractorFactoryMap.find(extractorType);
            if (extractorIt == extractorFactoryMap.end()) {
                LOG_ERROR("Unsupported feature extractor type: {}", extractorType);
                throw std::runtime_error("Unsupported feature extractor type: " + extractorType);
            }
            std::unique_ptr<FeatureExtractor> extractor = extractorIt->second();
            LOG_TRACE("Using {} feature extractor", extractorType);

            // Create matcher
            auto matcherIt = matcherFactoryMap.find(matcherType);
            if (matcherIt == matcherFactoryMap.end()) {
                LOG_ERROR("Unsupported feature matcher type: {}", matcherType);
                throw std::runtime_error("Unsupported feature matcher type: " + matcherType);
            }
            auto matcher = matcherIt->second();
            LOG_TRACE("Using {} feature matcher", matcherType);

            // Extract keypoints and descriptors in parallel
            auto extractFuture1 = std::async(std::launch::async, [&extractor, &image1]() {
                return extractor->extract(image1);
            });
            auto extractFuture2 = std::async(std::launch::async, [&extractor, &image2]() {
                return extractor->extract(image2);
            });

            // Extract keypoints and descriptors
            auto [keypoints1, descriptors1] = extractFuture1.get();
            auto [keypoints2, descriptors2] = extractFuture2.get();

            LOG_DEBUG("Image 1: {} descriptors and {} keypoints.", descriptors1.rows, keypoints1.size());
            LOG_DEBUG("Image 2: {} descriptors and {} keypoints.", descriptors2.rows, keypoints2.size());

            if (descriptors1.empty() || descriptors2.empty()) {
                LOG_ERROR("Descriptors extraction failed for one or both images");
                return 0.0;
            }

            std::vector<std::vector<cv::DMatch> > knnMatches;
            matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

            if (knnMatches.empty()) {
                LOG_ERROR("KNN matching failed to find any matches");
                return 0.0;
            }

            constexpr float ratioThresh = 0.75f;
            std::vector<cv::DMatch> goodMatches;
            for (const auto &knnMatch: knnMatches) {
                if (knnMatch[0].distance < ratioThresh * knnMatch[1].distance) {
                    goodMatches.push_back(knnMatch[0]);
                }
            }

            if (goodMatches.empty()) {
                LOG_ERROR("No good matches found after applying ratio test");
                return 0.0;
            }

            std::vector<cv::Point2f> points1, points2;
            for (const auto &match: goodMatches) {
                points1.push_back(keypoints1[match.queryIdx].pt);
                points2.push_back(keypoints2[match.trainIdx].pt);
            }

            std::vector<unsigned char> inliersMask;
            if (points1.size() >= 4 && points2.size() >= 4) {
                cv::findHomography(points1, points2, cv::RANSAC, 3, inliersMask);
            } else {
                inliersMask.resize(points1.size(), 0);
            }

            size_t numInliers = std::count(inliersMask.begin(), inliersMask.end(), 1);
            int numMatches = static_cast<int>(goodMatches.size());
            double score = numMatches > 0 ? static_cast<double>(numInliers) / numMatches : 0.0;

            LOG_DEBUG("Found {} inliers, {} good matches; similarity score = {}.", numInliers, goodMatches.size(),
                      score);

            return score;
        } catch (const std::exception &e) {
            LOG_ERROR("Exception during image comparison: {}", e.what());
            return 0.0;
        } catch (...) {
            LOG_ERROR("Unknown error during image comparison");
            return 0.0;
        }
    }*/
    double FeatureComparator::compare(const cv::Mat &image1, const cv::Mat &image2) const {
        // Check if images are empty
        if (image1.empty() || image2.empty()) {
            LOG_ERROR("One or both images are empty");
            throw std::invalid_argument("One or both images are empty");
        }

        // Use SIFT for feature extraction
        auto extractor = cv::SIFT::create();

        // Detect keypoints and compute descriptors
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;

        extractor->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
        extractor->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

        LOG_DEBUG("Image 1: {} descriptors and {} keypoints.", descriptors1.rows, keypoints1.size());
        LOG_DEBUG("Image 2: {} descriptors and {} keypoints.", descriptors2.rows, keypoints2.size());

        // Check if descriptors are empty
        if (descriptors1.empty() || descriptors2.empty()) {
            LOG_ERROR("Descriptors extraction failed for one or both images");
            return 0.0;
        }

        // Use BFMatcher with default parameters
        cv::BFMatcher matcher(cv::NORM_L2);
        std::vector<std::vector<cv::DMatch> > knnMatches;
        matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

        if (knnMatches.empty()) {
            LOG_ERROR("KNN matching failed to find any matches");
            return 0.0;
        }

        // Apply ratio test to find good matches
        constexpr float ratioThresh = 0.75f;
        std::vector<cv::DMatch> goodMatches;
        for (const auto &knnMatch: knnMatches) {
            if (knnMatch[0].distance < ratioThresh * knnMatch[1].distance) {
                goodMatches.push_back(knnMatch[0]);
            }
        }

        LOG_DEBUG("Found {} good matches.", goodMatches.size());

        // Calculate similarity score based on the number of good matches
        double score = static_cast<double>(goodMatches.size()) / std::min(keypoints1.size(), keypoints2.size());

        LOG_DEBUG("Similarity score = {}.", score);

        return score;
    }
}
