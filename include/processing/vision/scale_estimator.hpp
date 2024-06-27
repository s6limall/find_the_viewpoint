// File: processing/vision/scale_estimator.hpp

#ifndef SCALE_ESTIMATOR_HPP
#define SCALE_ESTIMATOR_HPP

#include <opencv2/opencv.hpp>
#include <memory>

#include "processing/vision/object_detector.hpp"

namespace processing::vision {
    class ScaleEstimator {
    public:
        explicit ScaleEstimator(const cv::Mat& camera_matrix, std::shared_ptr<ObjectDetector> detector);

        // Detects the object and estimates the scale and distance
        std::pair<float, float> estimateScaleAndDistance(const cv::Mat& target_image) const;

    private:
        cv::Mat camera_matrix_;
        std::shared_ptr<ObjectDetector> detector_;
    };
}

#endif // SCALE_ESTIMATOR_HPP
