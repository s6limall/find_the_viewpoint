// File: processing/image/comparator/feature_comparator.hpp

#ifndef FEATURE_COMPARATOR_HPP
#define FEATURE_COMPARATOR_HPP

#include <algorithm>
#include <cmath>
#include <execution>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

#include "processing/image/comparator.hpp"
#include "processing/image/feature/extractor.hpp"
#include "processing/image/feature/matcher.hpp"
#include "types/image.hpp"

namespace processing::image {

    class FeatureComparator final : public ImageComparator {
    public:
        FeatureComparator(std::shared_ptr<FeatureExtractor> extractor, std::shared_ptr<FeatureMatcher> matcher) noexcept
            : extractor_(std::move(extractor)), matcher_(std::move(matcher)) {}

        [[nodiscard]] double compare(const cv::Mat &image1, const cv::Mat &image2) const override {
            auto extracted1 = extractor_->extract(image1);
            auto extracted2 = extractor_->extract(image2);

            // Convert the extracted features to FeatureMatcher::Features
            const FeatureMatcher::Features features1{extracted1.second, extracted1.first};
            const FeatureMatcher::Features features2{extracted2.second, extracted2.first};

            return compareFeatures(features1, features2);
        }

        [[nodiscard]] double compare(const Image<> &img1, const Image<> &img2) const override {
            const FeatureMatcher::Features features1{img1.getDescriptors(), img1.getKeypoints()};
            const FeatureMatcher::Features features2{img2.getDescriptors(), img2.getKeypoints()};
            return compareFeatures(features1, features2);
        }


    private:
        std::shared_ptr<FeatureExtractor> extractor_;
        std::shared_ptr<FeatureMatcher> matcher_;

        [[nodiscard]] double compareFeatures(const FeatureMatcher::Features &features1,
                                             const FeatureMatcher::Features &features2) const {
            const auto &[descriptors1, keypoints1] = features1;
            const auto &[descriptors2, keypoints2] = features2;

            if (descriptors1.empty() || descriptors2.empty()) {
                LOG_WARN("Descriptors are empty for one or both images");
                return 0.0;
            }

            auto [matches, homography, inliers] = matcher_->match(features1, features2);
            if (matches.empty())
                return 0.0;

            return calculateScore(matches, keypoints1, keypoints2, homography, inliers);
        }


        [[nodiscard]] static double calculateScore(const std::vector<cv::DMatch> &matches,
                                                   const std::vector<cv::KeyPoint> &keypoints1,
                                                   const std::vector<cv::KeyPoint> &keypoints2,
                                                   const cv::Mat &homography, const std::vector<uchar> &inliers) {
            const double match_score = calculateMatchScore(matches, keypoints1, keypoints2);
            const double inlier_score = calculateInlierScore(inliers, matches.size());
            const double distance_score = calculateDistanceScore(matches);
            const double perspective_score =
                    calculatePerspectiveScore(keypoints1, keypoints2, matches, homography, inliers);

            // Weighted combination of scores
            const double score =
                    0.3 * match_score + 0.3 * inlier_score + 0.2 * distance_score + 0.2 * perspective_score;

            LOG_INFO("Feature Comparator - Score: {:.4f} (Match: {:.4f}, Inlier: {:.4f}, Distance: {:.4f}, "
                     "Perspective: {:.4f}) - {} matches.",
                     score, match_score, inlier_score, distance_score, perspective_score, matches.size());

            return score;
        }

        [[nodiscard]] static double calculateMatchScore(const std::vector<cv::DMatch> &matches,
                                                        const std::vector<cv::KeyPoint> &keypoints1,
                                                        const std::vector<cv::KeyPoint> &keypoints2) {
            const size_t total_keypoints = std::max(keypoints1.size(), keypoints2.size());
            return static_cast<double>(matches.size()) / total_keypoints;
        }

        [[nodiscard]] static double calculateInlierScore(const std::vector<uchar> &inliers, size_t total_matches) {
            return static_cast<double>(cv::countNonZero(inliers)) / total_matches;
        }

        [[nodiscard]] static double calculateDistanceScore(const std::vector<cv::DMatch> &matches) {
            if (matches.empty())
                return 0.0;

            const auto [min_dist, max_dist] =
                    std::minmax_element(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
                        return a.distance < b.distance;
                    });

            if (min_dist->distance == max_dist->distance)
                return 1.0;

            const double distance_range = max_dist->distance - min_dist->distance;
            return std::transform_reduce(std::execution::par_unseq, matches.begin(), matches.end(), 0.0, std::plus<>(),
                                         [min_dist, distance_range](const cv::DMatch &match) {
                                             return 1.0 - (match.distance - min_dist->distance) / distance_range;
                                         }) /
                   matches.size();
        }

        [[nodiscard]] static double calculatePerspectiveScore(const std::vector<cv::KeyPoint> &keypoints1,
                                                              const std::vector<cv::KeyPoint> &keypoints2,
                                                              const std::vector<cv::DMatch> &matches,
                                                              const cv::Mat &homography,
                                                              const std::vector<uchar> &inliers) {
            if (homography.empty() || inliers.empty())
                return 0.0;

            std::vector<cv::Point2f> src_pts, dst_pts;
            src_pts.reserve(inliers.size());
            dst_pts.reserve(inliers.size());

            for (size_t i = 0; i < matches.size(); ++i) {
                if (inliers[i]) {
                    src_pts.push_back(keypoints1[matches[i].queryIdx].pt);
                    dst_pts.push_back(keypoints2[matches[i].trainIdx].pt);
                }
            }

            if (src_pts.empty())
                return 0.0;

            std::vector<cv::Point2f> transformed_pts;
            cv::perspectiveTransform(src_pts, transformed_pts, homography);

            const double max_dimension = std::max(keypoints1[0].pt.x, keypoints1[0].pt.y);
            const double normalized_error =
                    std::transform_reduce(std::execution::par_unseq, dst_pts.begin(), dst_pts.end(),
                                          transformed_pts.begin(), 0.0, std::plus<>(),
                                          [max_dimension](const cv::Point2f &p1, const cv::Point2f &p2) {
                                              return cv::norm(p1 - p2) / max_dimension;
                                          }) /
                    src_pts.size();

            return std::exp(-normalized_error);
        }
    };

} // namespace processing::image

#endif // FEATURE_COMPARATOR_HPP
