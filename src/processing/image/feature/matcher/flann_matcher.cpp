// File: processing/image/feature/matcher/flann.cpp

#include "processing/image/feature/matcher/flann_matcher.hpp"

#include "config/configuration.hpp"

#include <spdlog/spdlog.h>

namespace processing::image {
    cv::Mat FLANNMatcher::convertDescriptorsToFloat(const cv::Mat &descriptors) {
        cv::Mat float_descriptors;
        descriptors.convertTo(float_descriptors, CV_32F);
        return float_descriptors;
    }

    std::vector<cv::DMatch> FLANNMatcher::filterMatches(const std::vector<std::vector<cv::DMatch> > &knn_matches,
                                                        float ratio_thresh) {
        std::vector<cv::DMatch> good_matches;
        good_matches.reserve(knn_matches.size());
        for (const auto &knn_match: knn_matches) {
            if (knn_match.size() == 2 && knn_match[0].distance < ratio_thresh * knn_match[1].distance) {
                good_matches.push_back(knn_match[0]);
            }
        }
        return good_matches;
    }

    std::vector<cv::DMatch> FLANNMatcher::match(const cv::Mat &descriptors1, const cv::Mat &descriptors2) const {
        if (descriptors1.empty() || descriptors2.empty()) {
            spdlog::error("One or both descriptor matrices are empty");
            return {};
        }

        const auto desc1 = convertDescriptorsToFloat(descriptors1);
        const auto desc2 = convertDescriptorsToFloat(descriptors2);

        const auto ratio_thresh = config::get("feature_matcher.flann.ratio_thresh", 0.75f);
        const auto min_good_matches = config::get("feature_matcher.flann.min_good_matches", 10);

        cv::FlannBasedMatcher flann_matcher;
        std::vector<std::vector<cv::DMatch> > knn_matches;
        flann_matcher.knnMatch(desc1, desc2, knn_matches, 2);

        auto good_matches = filterMatches(knn_matches, ratio_thresh);

        if (good_matches.size() < min_good_matches) {
            spdlog::warn("Not enough good matches found: {}", good_matches.size());
        } else {
            spdlog::info("FLANN Matcher - {} good matches found", good_matches.size());
        }

        return good_matches;
    }

    void FLANNMatcher::knnMatch(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                                std::vector<std::vector<cv::DMatch> > &knnMatches, int k) const {
        if (descriptors1.empty() || descriptors2.empty()) {
            spdlog::error("One or both descriptor matrices are empty");
            return;
        }

        const auto desc1 = convertDescriptorsToFloat(descriptors1);
        const auto desc2 = convertDescriptorsToFloat(descriptors2);

        cv::FlannBasedMatcher matcher;
        matcher.knnMatch(desc1, desc2, knnMatches, k);
    }
}
