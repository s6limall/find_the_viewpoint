// File: processing/image/feature/matcher/flann_matcher.hpp

#ifndef FEATURE_MATCHER_FLANN_HPP
#define FEATURE_MATCHER_FLANN_HPP

#include <opencv2/calib3d.hpp>
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
            min_good_matches_(config::get("feature_matcher.flann.min_good_matches", 10)),
            min_homography_matches_(config::get("feature_matcher.flann.min_homography_matches", 4)) {

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
                LOG_WARN("One or both descriptor matrices are empty. desc1: {} x {}, desc2: {} x {}", desc1.rows,
                         desc1.cols, desc2.rows, desc2.cols);
                return {};
            }

            if (desc1.cols != desc2.cols) {
                LOG_ERROR("Descriptor dimensions do not match: {} vs {}", desc1.cols, desc2.cols);
                return {};
            }

            cv::Mat descriptors1, descriptors2;
            desc1.convertTo(descriptors1, CV_32F);
            desc2.convertTo(descriptors2, CV_32F);

            LOG_INFO("Matching descriptors: {} x {} vs {} x {}", descriptors1.rows, descriptors1.cols,
                     descriptors2.rows, descriptors2.cols);

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

        [[nodiscard]] double computeMatchScore(const std::vector<cv::KeyPoint> &keypoints1,
                                               const std::vector<cv::KeyPoint> &keypoints2,
                                               const std::vector<cv::DMatch> &matches) const {
            if (matches.size() < static_cast<size_t>(min_homography_matches_)) {
                // Fallback: use a simple ratio of good matches to total keypoints
                return static_cast<double>(matches.size()) / std::min(keypoints1.size(), keypoints2.size());
            }

            std::vector<cv::Point2f> points1, points2;
            for (const auto &match: matches) {
                points1.push_back(keypoints1[match.queryIdx].pt);
                points2.push_back(keypoints2[match.trainIdx].pt);
            }

            cv::Mat homography;
            std::vector<uchar> inliers;
            try {
                homography = cv::findHomography(points1, points2, cv::RANSAC, 3.0, inliers);
            } catch (const cv::Exception &e) {
                LOG_WARN("Homography calculation failed: {}. Falling back to simple match ratio.", e.what());
                return static_cast<double>(matches.size()) / std::min(keypoints1.size(), keypoints2.size());
            }

            if (homography.empty()) {
                LOG_WARN("Homography is empty. Falling back to simple match ratio.");
                return static_cast<double>(matches.size()) / std::min(keypoints1.size(), keypoints2.size());
            }

            int inlier_count = cv::countNonZero(inliers);
            double inlier_ratio = static_cast<double>(inlier_count) / matches.size();
            double match_ratio = static_cast<double>(matches.size()) / std::min(keypoints1.size(), keypoints2.size());

            // Combine inlier ratio and match ratio for a more robust score
            return 0.5 * (inlier_ratio + match_ratio);
        }

    private:
        cv::Ptr<cv::FlannBasedMatcher> matcher_;
        const float ratio_thresh_;
        const int min_good_matches_;
        const int min_homography_matches_;
    };

} // namespace processing::image

#endif // FEATURE_MATCHER_FLANN_HPP
