// File: processing/image/feature/matcher/flann.cpp

#include "processing/image/feature/matcher/flann_matcher.hpp"

#include "config/configuration.hpp"

#include <spdlog/spdlog.h>

namespace processing::image {
    std::vector<cv::DMatch> FLANNMatcher::match(const cv::Mat &descriptors1, const cv::Mat &descriptors2) const {
        // Check if descriptors are empty
        if (descriptors1.empty() || descriptors2.empty()) {
            spdlog::error("One or both descriptor matrices are empty");
            // throw std::invalid_argument("One or both descriptor matrices are empty");
        }

        // Convert descriptors to floating-point if necessary
        cv::Mat desc1, desc2;
        if (descriptors1.type() != CV_32F) {
            descriptors1.convertTo(desc1, CV_32F);
        } else {
            desc1 = descriptors1;
        }

        if (descriptors2.type() != CV_32F) {
            descriptors2.convertTo(desc2, CV_32F);
        } else {
            desc2 = descriptors2;
        }

        // Log configuration parameters
        const auto &config = config::Configuration::getInstance();
        float ratio_thresh = config.get<float>("feature_matcher.flann.ratio_thresh", 0.75f);
        int min_good_matches = config.get<int>("feature_matcher.flann.min_good_matches", 10);
        spdlog::debug("FLANN Matcher Configuration - Ratio Threshold: {}, Min Good Matches: {}", ratio_thresh,
                      min_good_matches);

        // Perform FLANN matching
        cv::FlannBasedMatcher flann_matcher;
        std::vector<std::vector<cv::DMatch> > knn_matches;
        flann_matcher.knnMatch(desc1, desc2, knn_matches, 2);
        spdlog::debug("FLANN Matcher - {} keypoints matched with {} keypoints", desc1.rows, desc2.rows);

        // Filter good matches based on ratio threshold
        std::vector<cv::DMatch> good_matches;
        for (const auto &knn_match: knn_matches) {
            if (knn_match.size() == 2 && knn_match[0].distance < ratio_thresh * knn_match[1].distance) {
                good_matches.push_back(knn_match[0]);
            }
        }

        // Check if enough good matches are found
        if (good_matches.size() < min_good_matches) {
            spdlog::warn("Not enough good matches found: {}", good_matches.size());
            // throw std::runtime_error("Not enough good matches found");
        }

        // Log number of good matches found
        spdlog::info("FLANN Matcher - {} good matches found", good_matches.size());

        return good_matches;
    }
}
