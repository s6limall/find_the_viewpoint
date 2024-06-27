// File: processing/vision/distance_estimator.cpp

#include <numeric>
#include <spdlog/spdlog.h>

#include "processing/vision/distance_estimator.hpp"
#include "processing/image/feature/extractor/orb_extractor.hpp"

namespace processing::vision {

    DistanceEstimator::DistanceEstimator(double unit_cube_size, int image_width, double fov_x) :
        unit_cube_size_(unit_cube_size),
        focal_length_(calculateFocalLength(image_width, fov_x)),
        feature_extractor_(image::FeatureExtractor::create<image::ORBExtractor>()) {
        spdlog::info("DistanceEstimator initialized with focal_length={}, unit_cube_size={}, image_width={}, fov_x={}",
                     focal_length_, unit_cube_size_, image_width, fov_x);
    }

    double DistanceEstimator::calculateFocalLength(int image_width, double fov_x) {
        // Convert fov_x to radians if it is in degrees
        double fov_x_rad = (fov_x > 0 && fov_x <= 2 * CV_PI) ? fov_x : (fov_x * CV_PI / 180.0);

        // Log the conversion
        if (fov_x != fov_x_rad) {
            spdlog::debug("Converting degrees to radians: {}", fov_x);
        }

        // Calculate the focal length
        double focal_length = image_width / (2.0 * std::tan(fov_x_rad / 2.0));
        spdlog::debug("Calculated focal length: {} using image_width={} and fov_x_rad={}", focal_length, image_width,
                      fov_x_rad);
        return focal_length;
    }

    double DistanceEstimator::calculateAverageKeypointSize(const std::vector<cv::KeyPoint> &keypoints) {
        if (keypoints.empty()) {
            spdlog::warn("No keypoints to calculate average size.");
            throw std::runtime_error("No keypoints to calculate average size.");
        }

        double avg_keypoint_size = std::accumulate(keypoints.begin(), keypoints.end(), 0.0,
                                                   [](double sum, const cv::KeyPoint &kp) {
                                                       return sum + kp.size;
                                                   }) / keypoints.size();
        spdlog::info("Average keypoint size: {}", avg_keypoint_size);
        return avg_keypoint_size;
    }

    double DistanceEstimator::estimate(const cv::Mat &image) {
        spdlog::info("Starting distance estimation.");

        if (image.empty()) {
            spdlog::error("Input image is empty.");
            throw std::invalid_argument("Input image is empty.");
        }

        auto [keypoints, descriptors] = feature_extractor_->extract(image);
        if (keypoints.empty()) {
            spdlog::warn("No keypoints detected in the image.");
            throw std::runtime_error("No keypoints detected in the image.");
        }

        double avg_keypoint_size = calculateAverageKeypointSize(keypoints);
        double distance = (unit_cube_size_ * focal_length_) / avg_keypoint_size;
        spdlog::info("Estimated distance to object: {}", distance);

        return distance;
    }

}
