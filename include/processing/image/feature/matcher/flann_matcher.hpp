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

        explicit FLANNMatcher(const IndexType index_type = IndexType::KDTree) noexcept :
            ratio_thresh_(config::get("feature_matcher.flann.ratio_thresh", 0.75f)),
            min_good_matches_(config::get("feature_matcher.flann.min_good_matches", 10)),
            min_homography_matches_(config::get("feature_matcher.flann.min_homography_matches", 4)) {

            if (index_type == IndexType::KDTree) {
                int trees = config::get("feature_matcher.flann.trees", 5);
                matcher_ = cv::makePtr<cv::FlannBasedMatcher>(cv::makePtr<cv::flann::KDTreeIndexParams>(trees));
            } else {
                int table_number = config::get("feature_matcher.flann.lsh_table_number", 12);
                int key_size = config::get("feature_matcher.flann.lsh_key_size", 20);
                int multi_probe_level = config::get("feature_matcher.flann.lsh_multi_probe_level", 2);
                matcher_ = cv::makePtr<cv::FlannBasedMatcher>(
                        cv::makePtr<cv::flann::LshIndexParams>(table_number, key_size, multi_probe_level));
            }
        }

        [[nodiscard]] MatchResult match(const Features &features1, const Features &features2) const override {
            const auto &[descriptors1, keypoints1] = features1;
            const auto &[descriptors2, keypoints2] = features2;

            if (descriptors1.empty() || descriptors2.empty()) {
                LOG_WARN("One or both descriptor matrices are empty. desc1: {} x {}, desc2: {} x {}", descriptors1.rows,
                         descriptors1.cols, descriptors2.rows, descriptors2.cols);
                return {};
            }

            if (descriptors1.cols != descriptors2.cols) {
                LOG_ERROR("Descriptor dimensions do not match: {} vs {}", descriptors1.cols, descriptors2.cols);
                return {};
            }

            cv::Mat desc1, desc2;
            descriptors1.convertTo(desc1, CV_32F);
            descriptors2.convertTo(desc2, CV_32F);

            LOG_INFO("Matching descriptors: {} x {} vs {} x {}", desc1.rows, desc1.cols, desc2.rows, desc2.cols);

            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher_->knnMatch(desc1, desc2, knn_matches, 2);

            std::vector<cv::DMatch> good_matches;
            for (const auto &match: knn_matches) {
                if (match.size() == 2 && match[0].distance < ratio_thresh_ * match[1].distance) {
                    good_matches.push_back(match[0]);
                }
            }

            if (good_matches.size() < static_cast<size_t>(min_good_matches_)) {
                LOG_WARN("Not enough good matches found: {}", good_matches.size());
                return {good_matches, {}, {}};
            }

            if (good_matches.size() < static_cast<size_t>(min_homography_matches_)) {
                LOG_WARN("Too few matches for homography: {}", good_matches.size());
                return {good_matches, {}, {}};
            }

            std::vector<cv::Point2f> points1, points2;
            for (const auto &match: good_matches) {
                points1.push_back(keypoints1[match.queryIdx].pt);
                points2.push_back(keypoints2[match.trainIdx].pt);
            }

            cv::Mat homography;
            std::vector<uchar> inliers;
            try {
                homography = cv::findHomography(points1, points2, cv::RANSAC, 3.0, inliers);
            } catch (const cv::Exception &e) {
                LOG_WARN("Homography calculation failed: {}. Falling back to simple match ratio.", e.what());
                return {good_matches, {}, {}};
            }

            if (homography.empty()) {
                LOG_WARN("Homography is empty. Falling back to simple match ratio.");
                return {good_matches, {}, {}};
            }

            return {good_matches, homography, inliers};
        }

    private:
        cv::Ptr<cv::FlannBasedMatcher> matcher_;
        const float ratio_thresh_;
        const int min_good_matches_;
        const int min_homography_matches_;
    };

} // namespace processing::image

#endif // FEATURE_MATCHER_FLANN_HPP
