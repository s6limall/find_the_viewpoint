// File: processing/image/feature/matcher/flann.cpp

#include "processing/image/feature/matcher/flann_matcher.hpp"

namespace processing::image {

    cv::Mat FLANNMatcher::convertDescriptorsToFloat(const cv::Mat &descriptors) noexcept {
        cv::Mat floatDescriptors;
        descriptors.convertTo(floatDescriptors, CV_32F);
        return floatDescriptors;
    }

    // Ratio Test
    std::vector<cv::DMatch> FLANNMatcher::filterMatches(const std::vector<std::vector<cv::DMatch>> &knnMatches,
                                                        const float ratioThresh) noexcept {
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
        const int trees =
                std::min(5, std::max(1, desc1.rows / 1000)); // Adapt the number of trees based on the dataset size
        const cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(trees);
        const cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>();
        const cv::FlannBasedMatcher matcher(indexParams, searchParams);

        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(desc1, desc2, knnMatches, k);
        return knnMatches;
    }

    std::vector<cv::DMatch> FLANNMatcher::match(const cv::Mat &descriptors1, const cv::Mat &descriptors2) const {
        if (descriptors1.empty() || descriptors2.empty()) {
            LOG_ERROR("One or both descriptor matrices are empty");
            throw std::invalid_argument("One or both descriptor matrices are empty");
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
            LOG_INFO("FLANN Matcher - {} good matches found", goodMatches.size());
        }

        return goodMatches;
    }
} // namespace processing::image
