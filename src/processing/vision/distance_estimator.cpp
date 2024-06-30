// File: processing/vision/distance_estimator.cpp

#include <numeric>
#include "common/logging/logger.hpp"
#include "processing/vision/distance_estimator.hpp"
#include "common/utilities/camera_utils.hpp"
#include "processing/image/feature/extractor/orb_extractor.hpp"

namespace processing::vision {

    DistanceEstimator::DistanceEstimator(const float focal_length, const float unit_cube_size) :
        focal_length_(focal_length),
        unit_cube_size_(unit_cube_size),
        feature_extractor_(image::FeatureExtractor::create<image::ORBExtractor>()) {
        LOG_INFO("DistanceEstimator initialized with focal_length={}, unit_cube_size={}", focal_length_,
                 unit_cube_size_);
    }

    double DistanceEstimator::calculateAverageKeypointSize(const std::vector<cv::KeyPoint> &keypoints) {
        if (keypoints.empty()) {
            LOG_WARN("No keypoints to calculate average size.");
            throw std::runtime_error("No keypoints to calculate average size.");
        }

        double avg_keypoint_size = std::accumulate(keypoints.begin(), keypoints.end(), 0.0,
                                                   [](double sum, const cv::KeyPoint &kp) {
                                                       return sum + kp.size;
                                                   }) / static_cast<double>(keypoints.size());
        LOG_INFO("Calculated average keypoint size: {}", avg_keypoint_size);
        return avg_keypoint_size;
    }

    double DistanceEstimator::estimate(const cv::Mat &image) {
        LOG_INFO("Starting distance estimation.");

        if (image.empty()) {
            LOG_ERROR("Input image is empty.");
            throw std::invalid_argument("Input image is empty.");
        }

        auto [keypoints, descriptors] = feature_extractor_->extract(image);
        if (keypoints.empty()) {
            LOG_WARN("No keypoints detected in the image.");
            throw std::runtime_error("No keypoints detected in the image.");
        }

        const double avg_keypoint_size = calculateAverageKeypointSize(keypoints);

        if (avg_keypoint_size == 0.0) {
            LOG_ERROR("Average keypoint size is zero, cannot estimate distance.");
            throw std::runtime_error("Average keypoint size is zero, cannot estimate distance.");
        }

        double distance = (unit_cube_size_ * focal_length_) / avg_keypoint_size;
        LOG_INFO("Estimated distance to object: {} (unit_cube_size: {}, focal_length: {}, avg_keypoint_size: {})",
                 distance, unit_cube_size_, focal_length_, avg_keypoint_size);

        return distance;
    }

}
