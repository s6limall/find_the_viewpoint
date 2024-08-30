// File: processing/image/comparator/feature_comparator.hpp

#ifndef FEATURE_COMPARATOR_HPP
#define FEATURE_COMPARATOR_HPP

#include <algorithm>
#include <cmath>
#include <execution>
#include <memory>
#include <numeric>
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
            const auto features1 = extractor_->extract(image1);
            const auto features2 = extractor_->extract(image2);
            return compareFeatures(FeatureMatcher::Features{features1.second, features1.first},
                                   FeatureMatcher::Features{features2.second, features2.first});
        }

        [[nodiscard]] double compare(const Image<> &img1, const Image<> &img2) const override {
            return compareFeatures(FeatureMatcher::Features{img1.getDescriptors(), img1.getKeypoints()},
                                   FeatureMatcher::Features{img2.getDescriptors(), img2.getKeypoints()});
        }

    private:
        std::shared_ptr<FeatureExtractor> extractor_;
        std::shared_ptr<FeatureMatcher> matcher_;

        [[nodiscard]] double compareFeatures(const FeatureMatcher::Features &features1,
                                             const FeatureMatcher::Features &features2) const {
            if (features1.first.empty() || features2.first.empty()) {
                LOG_WARN("Descriptors are empty for one or both images");
                return 0.0;
            }

            auto [matches, homography, inliers] = matcher_->match(features1, features2);
            LOG_DEBUG("Found {} matches", matches.size());

            if (matches.empty()) {
                return 0.0;
            }

            return calculateScore(matches, features1.second.size(), features2.second.size(), homography, inliers,
                                  features1.second, features2.second);
        }

        [[nodiscard]] static double calculateScore(const std::vector<cv::DMatch> &matches, const size_t keypoints1_size,
                                                   const size_t keypoints2_size, const cv::Mat &homography,
                                                   const std::vector<uchar> &inliers,
                                                   const std::vector<cv::KeyPoint> &keypoints1,
                                                   const std::vector<cv::KeyPoint> &keypoints2) {
            const double match_ratio = static_cast<double>(matches.size()) / std::min(keypoints1_size, keypoints2_size);
            const double inlier_ratio = static_cast<double>(cv::countNonZero(inliers)) / matches.size();

            const double distance_score = calculateDistanceScore(matches);
            const double perspective_score =
                    calculatePerspectiveScore(keypoints1, keypoints2, matches, homography, inliers);

            // Weighted combination of scores
            const double score =
                    0.4 * match_ratio + 0.3 * inlier_ratio + 0.2 * distance_score + 0.1 * perspective_score;

            LOG_INFO("Feature Comparator - Score: {:.4f} (Match Ratio: {:.4f}, Inlier Ratio: {:.4f}, "
                     "Distance Score: {:.4f}, Perspective Score: {:.4f}) - {} matches.",
                     score, match_ratio, inlier_ratio, distance_score, perspective_score, matches.size());

            return score;
        }

        [[nodiscard]] static double calculateDistanceScore(const std::vector<cv::DMatch> &matches) {
            if (matches.empty()) {
                return 0.0;
            }

            const auto [min_dist, max_dist] =
                    std::minmax_element(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
                        return a.distance < b.distance;
                    });

            if (min_dist->distance == max_dist->distance) {
                return 1.0; // All distances are equal, consider it a perfect match
            }

            double sum_normalized_distances = 0.0;
            for (const auto &match: matches) {
                double normalized_distance =
                        (match.distance - min_dist->distance) / (max_dist->distance - min_dist->distance);
                sum_normalized_distances += 1.0 - normalized_distance;
            }

            return sum_normalized_distances / matches.size();
        }

        [[nodiscard]] static double calculatePerspectiveScore(const std::vector<cv::KeyPoint> &keypoints1,
                                                              const std::vector<cv::KeyPoint> &keypoints2,
                                                              const std::vector<cv::DMatch> &matches,
                                                              const cv::Mat &homography,
                                                              const std::vector<uchar> &inliers) {
            if (homography.empty() || inliers.empty()) {
                return 0.0;
            }

            double total_error = 0.0;
            int inlier_count = 0;

            for (size_t i = 0; i < matches.size(); ++i) {
                if (inliers[i]) {
                    const cv::Point2f &pt1 = keypoints1[matches[i].queryIdx].pt;
                    const cv::Point2f &pt2 = keypoints2[matches[i].trainIdx].pt;

                    cv::Mat pt1_homogeneous = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1.0);
                    cv::Mat transformed_pt = homography * pt1_homogeneous;
                    cv::Point2f transformed_pt2(transformed_pt.at<double>(0) / transformed_pt.at<double>(2),
                                                transformed_pt.at<double>(1) / transformed_pt.at<double>(2));

                    double error = cv::norm(pt2 - transformed_pt2);
                    total_error += error;
                    ++inlier_count;
                }
            }

            if (inlier_count == 0) {
                return 0.0;
            }

            double avg_error = total_error / inlier_count;
            return std::exp(-avg_error / 10.0); // Exponential decay of error
        }
    };

} // namespace processing::image

#endif // FEATURE_COMPARATOR_HPP
