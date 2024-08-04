// File: processing/image/comparator/feature_comparator.hpp

#ifndef FEATURE_COMPARATOR_HPP
#define FEATURE_COMPARATOR_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

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
            return compareFeatures(extractor_->extract(image1), extractor_->extract(image2));
        }

        [[nodiscard]] double compare(const Image<> &img1, const Image<> &img2) const override {
            LOG_DEBUG("Image 1: {} x {}; KeyPoints: {}, Descriptors: {}", img1.getImage().cols, img1.getImage().rows,
                      img1.getKeypoints().size(), img1.getDescriptors().rows);
            LOG_DEBUG("Image 2: {} x {}; KeyPoints: {}, Descriptors: {}", img2.getImage().cols, img2.getImage().rows,
                      img2.getKeypoints().size(), img2.getDescriptors().rows);
            return compareFeatures({img1.getKeypoints(), img1.getDescriptors()},
                                   {img2.getKeypoints(), img2.getDescriptors()});
        }


    private:
        std::shared_ptr<FeatureExtractor> extractor_;
        std::shared_ptr<FeatureMatcher> matcher_;

        using FeatureSet = std::pair<std::vector<cv::KeyPoint>, cv::Mat>;

        [[nodiscard]] double compareFeatures(const FeatureSet &features1, const FeatureSet &features2) const {
            const auto &[keypoints1, descriptors1] = features1;
            const auto &[keypoints2, descriptors2] = features2;

            if (descriptors1.empty() || descriptors2.empty()) {
                LOG_WARN("Descriptors are empty for one or both images");
                return 0.0;
            }

            const auto matches = matcher_->match(descriptors1, descriptors2);

            if (matches.empty()) {
                LOG_WARN("No matches found.");
                return 0.0;
            }

            // If we have very few matches, we skip the homography calculation as it requires 4+
            if (matches.size() < 4) {
                LOG_WARN("Too few matches ({}) to calculate homography. Using simple match ratio.", matches.size());
                return static_cast<double>(matches.size()) / std::min(keypoints1.size(), keypoints2.size());
            }

            const auto [inlier_matches, homography] = findInliers(keypoints1, keypoints2, matches);

            if (inlier_matches.empty()) {
                LOG_WARN("No inlier matches found. Using all matches for score calculation.");
                return static_cast<double>(matches.size()) / std::min(keypoints1.size(), keypoints2.size());
            }

            const double match_ratio =
                    static_cast<double>(inlier_matches.size()) / std::min(keypoints1.size(), keypoints2.size());

            const double avg_distance =
                    std::transform_reduce(inlier_matches.begin(), inlier_matches.end(), 0.0, std::plus<>(),
                                          [](const cv::DMatch &match) { return match.distance; }) /
                    inlier_matches.size();

            constexpr double max_distance = 128.0 * std::sqrt(2.0);
            const double normalized_distance = 1.0 - std::min(avg_distance / max_distance, 1.0);

            const double perspective_error =
                    calculatePerspectiveError(keypoints1, keypoints2, inlier_matches, homography);
            const double normalized_perspective_error = std::exp(-perspective_error / 10.0);

            const double score = match_ratio * 0.4 + normalized_distance * 0.4 + normalized_perspective_error * 0.2;

            LOG_INFO("Feature Comparator - Score: {:.4f} (Match Ratio: {:.4f}, Norm Distance: {:.4f}, Perspective "
                     "Error: {:.4f})",
                     score, match_ratio, normalized_distance, perspective_error);

            return score;
        }

        [[nodiscard]] static std::pair<std::vector<cv::DMatch>, cv::Mat>
        findInliers(const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2,
                    const std::vector<cv::DMatch> &matches) {
            std::vector<cv::Point2f> points1, points2;
            points1.reserve(matches.size());
            points2.reserve(matches.size());

            for (const auto &match: matches) {
                points1.push_back(keypoints1[match.queryIdx].pt);
                points2.push_back(keypoints2[match.trainIdx].pt);
            }

            std::vector<uchar> inlier_mask;
            cv::Mat homography = cv::findHomography(points1, points2, cv::RANSAC, 3.0, inlier_mask);

            std::vector<cv::DMatch> inlier_matches;
            inlier_matches.reserve(matches.size());
            for (size_t i = 0; i < inlier_mask.size(); ++i) {
                if (inlier_mask[i]) {
                    inlier_matches.push_back(matches[i]);
                }
            }

            return {inlier_matches, homography};
        }

        [[nodiscard]] static double calculatePerspectiveError(const std::vector<cv::KeyPoint> &keypoints1,
                                                              const std::vector<cv::KeyPoint> &keypoints2,
                                                              const std::vector<cv::DMatch> &matches,
                                                              const cv::Mat &homography) {
            double total_error = 0.0;
            for (const auto &match: matches) {
                cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
                cv::Point2f pt2 = keypoints2[match.trainIdx].pt;

                cv::Mat pt1_homogeneous = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1.0);
                cv::Mat transformed_pt = homography * pt1_homogeneous;
                cv::Point2f transformed_pt2(transformed_pt.at<double>(0) / transformed_pt.at<double>(2),
                                            transformed_pt.at<double>(1) / transformed_pt.at<double>(2));

                total_error += cv::norm(pt2 - transformed_pt2);
            }
            return total_error / matches.size();
        }
    };

} // namespace processing::image

#endif // FEATURE_COMPARATOR_HPP
