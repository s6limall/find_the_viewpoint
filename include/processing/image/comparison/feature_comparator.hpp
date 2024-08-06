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
            auto features1 = extractor_->extract(image1);
            auto features2 = extractor_->extract(image2);
            return compareFeatures(FeatureMatcher::Features{features1.second, features1.first},
                                   FeatureMatcher::Features{features2.second, features2.first});
        }

        [[nodiscard]] double compare(const Image<> &img1, const Image<> &img2) const override {
            LOG_DEBUG("Image 1: {} x {}; KeyPoints: {}, Descriptors: {}", img1.getImage().cols, img1.getImage().rows,
                      img1.getKeypoints().size(), img1.getDescriptors().rows);
            LOG_DEBUG("Image 2: {} x {}; KeyPoints: {}, Descriptors: {}", img2.getImage().cols, img2.getImage().rows,
                      img2.getKeypoints().size(), img2.getDescriptors().rows);
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
            LOG_DEBUG("Found {} good matches", matches.size());

            if (matches.empty()) {
                return 0.0;
            }

            if (inliers.empty()) {
                return static_cast<double>(matches.size()) / std::min(features1.second.size(), features2.second.size());
            }

            return calculateScore(matches, features1.second.size(), features2.second.size(), homography, inliers,
                                  features1.second, features2.second);
        }

        [[nodiscard]] double calculateScore(const std::vector<cv::DMatch> &matches, const size_t keypoints1_size,
                                            const size_t keypoints2_size, const cv::Mat &homography,
                                            const std::vector<uchar> &inliers,
                                            const std::vector<cv::KeyPoint> &keypoints1,
                                            const std::vector<cv::KeyPoint> &keypoints2) const {

            const double match_ratio = static_cast<double>(matches.size()) / std::min(keypoints1_size, keypoints2_size);
            const double avg_distance =
                    std::transform_reduce(std::execution::par, matches.begin(), matches.end(), 0.0, std::plus<>(),
                                          [](const cv::DMatch &match) { return static_cast<double>(match.distance); }) /
                    matches.size();
            constexpr double max_distance = 128.0 * std::sqrt(2.0);
            const double normalized_distance = 1.0 - std::min(avg_distance / max_distance, 1.0);
            const double perspective_error =
                    calculatePerspectiveError(keypoints1, keypoints2, matches, homography, inliers);
            const double normalized_perspective_error = std::exp(-perspective_error / 10.0);

            const double score = match_ratio * 0.5 + normalized_distance * 0.3 + normalized_perspective_error * 0.2;
            LOG_INFO(
                    "Feature Comparator - Score: {:.4f} (Match Ratio: {:.4f}, Normalized Distance: {:.4f}, Perspective "
                    "Error: {:.4f}) - {} matches.",
                    score, match_ratio, normalized_distance, normalized_perspective_error, matches.size());

            return score;
        }

        [[nodiscard]] static double calculatePerspectiveError(const std::vector<cv::KeyPoint> &keypoints1,
                                                              const std::vector<cv::KeyPoint> &keypoints2,
                                                              const std::vector<cv::DMatch> &matches,
                                                              const cv::Mat &homography,
                                                              const std::vector<uchar> &inliers) {
            double total_error =
                    std::transform_reduce(std::execution::par, matches.begin(), matches.end(), 0.0, std::plus<>(),
                                          [&](const cv::DMatch &match) {
                                              if (!inliers[&match - &matches[0]]) {
                                                  return 0.0;
                                              }
                                              cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
                                              cv::Point2f pt2 = keypoints2[match.trainIdx].pt;

                                              cv::Mat pt1_homogeneous = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1.0);
                                              cv::Mat transformed_pt = homography * pt1_homogeneous;
                                              cv::Point2f transformed_pt2(
                                                      transformed_pt.at<double>(0) / transformed_pt.at<double>(2),
                                                      transformed_pt.at<double>(1) / transformed_pt.at<double>(2));
                                              return cv::norm(pt2 - transformed_pt2);
                                          }) /
                    cv::countNonZero(inliers);
            LOG_DEBUG("Total perspective error: {:.4f}", total_error);
            return total_error;
        }
    };

} // namespace processing::image

#endif // FEATURE_COMPARATOR_HPP
