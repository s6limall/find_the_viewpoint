// File: processing/vision/scale_estimator.cpp

#include <spdlog/spdlog.h>
#include "processing/vision/scale_estimator.hpp"

namespace processing::vision {

    ScaleEstimator::ScaleEstimator(const cv::Mat& camera_matrix, std::shared_ptr<ObjectDetector> detector)
        : camera_matrix_(camera_matrix), detector_(detector) {}

    std::pair<float, float> ScaleEstimator::estimateScaleAndDistance(const cv::Mat& image) const {
        auto [center, radius] = detector_->detect(image);

        if (radius <= 0) {
            spdlog::error("Invalid radius detected: {}", radius);
            return {0, 0};
        }

        float fx = static_cast<float>(camera_matrix_.at<double>(0, 0));
        float fy = static_cast<float>(camera_matrix_.at<double>(1, 1));

        if (fx <= 0 || fy <= 0) {
            spdlog::error("Invalid camera intrinsics: fx={}, fy={}", fx, fy);
            return {0, 0};
        }

        // Use fx and fy separately to estimate distance
        float distance_x = (fx * 1.0f) / (2.0f * radius);
        float distance_y = (fy * 1.0f) / (2.0f * radius);

        spdlog::debug("Estimated distances: distance_x={}, distance_y={}", distance_x, distance_y);

        // Average the distance estimations
        float distance = (distance_x + distance_y) / 2.0f;

        spdlog::info("Estimated distance: {}, scale: {}", distance, radius);
        return {radius, distance};
    }
}
