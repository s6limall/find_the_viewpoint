// File: processing/image/feature/matcher/bf.cpp

#include "processing/image/feature/matcher/bf_matcher.hpp"

#include "config/configuration.hpp"

namespace processing::image {
    std::vector<cv::DMatch> BFMatcher::match(const cv::Mat &descriptors1, const cv::Mat &descriptors2) const {
        if (descriptors1.empty() || descriptors2.empty()) {
            throw std::invalid_argument("One or both descriptor matrices are empty");
        }

        // Determine the norm type based on the descriptor type
        int norm_type;
        if (descriptors1.type() == CV_8U) {
            norm_type = cv::NORM_HAMMING;
        } else if (descriptors1.type() == CV_32F) {
            norm_type = cv::NORM_L2;
        } else {
            throw std::runtime_error("Unsupported descriptor type");
        }

        const auto &config = config::Configuration::getInstance();
        bool cross_check = config.get<bool>("feature_matcher.bf.cross_check", false);

        cv::BFMatcher bf_matcher(norm_type, cross_check);

        std::vector<cv::DMatch> matches;
        bf_matcher.match(descriptors1, descriptors2, matches);

        return matches;
    }

    void BFMatcher::knnMatch(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                             std::vector<std::vector<cv::DMatch> > &knnMatches, int k) const {
        if (descriptors1.empty() || descriptors2.empty()) {
            throw std::invalid_argument("One or both descriptor matrices are empty");
        }

        // Determine the norm type based on the descriptor type
        int norm_type;
        if (descriptors1.type() == CV_8U) {
            norm_type = cv::NORM_HAMMING;
        } else if (descriptors1.type() == CV_32F) {
            norm_type = cv::NORM_L2;
        } else {
            throw std::runtime_error("Unsupported descriptor type");
        }

        const auto &config = config::Configuration::getInstance();
        bool cross_check = config.get<bool>("feature_matcher.bf.cross_check", false);

        cv::BFMatcher bf_matcher(norm_type, cross_check);
        bf_matcher.knnMatch(descriptors1, descriptors2, knnMatches, k);
    }
}


