// File: processing/image/feature/matcher/flann_matcher.cpp

#include "processing/image/feature/matcher/flann_matcher.hpp"
#include "common/logging/logger.hpp"

// TODO: FIX THIS
namespace processing::image {

    cv::Mat FLANNMatcher::convertDescriptorsToFloat(const cv::Mat &descriptors) {
        cv::Mat floatDescriptors;
        if (descriptors.type() == CV_32F) {
            return descriptors;
        }
        descriptors.convertTo(floatDescriptors, CV_32F);
        return floatDescriptors;
    }

    std::vector<cv::DMatch> FLANNMatcher::filterMatches(const std::vector<std::vector<cv::DMatch>> &knnMatches,
                                                        const float ratioThresh) {
        std::vector<cv::DMatch> goodMatches;
        goodMatches.reserve(knnMatches.size());
        for (const auto &knnMatch: knnMatches) {
            if (knnMatch.size() == 2 && knnMatch[0].distance < ratioThresh * knnMatch[1].distance) {
                goodMatches.push_back(knnMatch[0]);
            }
        }
        return goodMatches;
    }

    std::vector<std::vector<cv::DMatch>> FLANNMatcher::knnMatch(const cv::Mat &desc1, const cv::Mat &desc2, int k) {
        // Adaptive parameters based on the number of descriptors
        const int trees = std::max(1, static_cast<int>(std::log(desc1.rows + 1)));
        const int checks = std::max(1, static_cast<int>(std::sqrt(desc1.rows + 1)));

        const cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(trees);
        const cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(checks);
        const cv::FlannBasedMatcher matcher(indexParams, searchParams);

        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(desc1, desc2, knnMatches, k);
        return knnMatches;
    }

    std::vector<cv::DMatch> FLANNMatcher::match(const cv::Mat &descriptors1, const cv::Mat &descriptors2) const {
        if (descriptors1.empty() || descriptors2.empty()) {
            LOG_ERROR("One or both sets of descriptors are empty.");
            throw std::invalid_argument("Empty descriptors provided.");
        }

        const auto desc1 = convertDescriptorsToFloat(descriptors1);
        const auto desc2 = convertDescriptorsToFloat(descriptors2);

        const float ratioThresh = config::get("feature_matcher.flann.ratio_thresh", 0.75f);
        const int minGoodMatches = config::get("feature_matcher.flann.min_good_matches", 10);

        const auto knnMatches = knnMatch(desc1, desc2, 2);
        auto goodMatches = filterMatches(knnMatches, ratioThresh);

        if (goodMatches.size() < static_cast<size_t>(minGoodMatches)) {
            LOG_WARN("Not enough good matches found: {}", goodMatches.size());
        } else {
            LOG_TRACE("FLANN Matcher - {} good matches found", goodMatches.size());
        }

        return goodMatches;
    }
} // namespace processing::image
