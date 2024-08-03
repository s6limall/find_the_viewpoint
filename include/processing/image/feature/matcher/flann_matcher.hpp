// File: processing/image/feature/matcher/flann_matcher.hpp

#ifndef FEATURE_MATCHER_FLANN_HPP
#define FEATURE_MATCHER_FLANN_HPP

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "config/configuration.hpp"
#include "processing/image/feature/matcher.hpp"

namespace processing::image {

    class FLANNMatcher final : public FeatureMatcher {
    public:
        enum class IndexType { KDTree, LSH };

        explicit FLANNMatcher(IndexType index_type = IndexType::KDTree) noexcept :
            ratio_thresh_(config::get("feature_matcher.flann.ratio_thresh", 0.75f)),
            min_good_matches_(config::get("feature_matcher.flann.min_good_matches", 10)) {

            if (index_type == IndexType::KDTree) {
                int trees = config::get("feature_matcher.flann.trees", 5);
                auto index_params = cv::makePtr<cv::flann::KDTreeIndexParams>(trees);
                matcher_ = cv::makePtr<cv::FlannBasedMatcher>(index_params);
            } else { // LSH
                int table_number = config::get("feature_matcher.flann.lsh_table_number", 12);
                int key_size = config::get("feature_matcher.flann.lsh_key_size", 20);
                int multi_probe_level = config::get("feature_matcher.flann.lsh_multi_probe_level", 2);
                auto index_params = cv::makePtr<cv::flann::LshIndexParams>(table_number, key_size, multi_probe_level);
                matcher_ = cv::makePtr<cv::FlannBasedMatcher>(index_params);
            }
        }

        [[nodiscard]] std::vector<cv::DMatch> match(const cv::Mat &desc1, const cv::Mat &desc2) const override {

            if (desc1.empty() || desc2.empty()) {
                LOG_ERROR("One or both descriptor matrices are empty");
                throw std::invalid_argument("One or both descriptor matrices are empty");
            }

            cv::Mat descriptors1, descriptors2;
            // Convert descriptors to CV_32F if they're not already
            desc1.convertTo(descriptors1, CV_32F);
            desc2.convertTo(descriptors2, CV_32F);

            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher_->knnMatch(descriptors1, descriptors2, knn_matches, 2);

            std::vector<cv::DMatch> good_matches;
            good_matches.reserve(knn_matches.size());

            for (const auto &match: knn_matches) {
                if (match.size() == 2 && match[0].distance < ratio_thresh_ * match[1].distance) {
                    good_matches.push_back(match[0]);
                }
            }

            if (good_matches.size() < static_cast<size_t>(min_good_matches_)) {
                LOG_WARN("Not enough good matches found: {}", good_matches.size());
            } else {
                LOG_INFO("FLANN Matcher - {} good matches found", good_matches.size());
            }

            return good_matches;
        }

    private:
        cv::Ptr<cv::FlannBasedMatcher> matcher_;
        const float ratio_thresh_;
        const int min_good_matches_;
    };

} // namespace processing::image

#endif // FEATURE_MATCHER_FLANN_HPP
