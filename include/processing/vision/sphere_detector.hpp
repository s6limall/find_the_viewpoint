// File: processing/vision/sphere_detector.hpp

#ifndef SPHERE_DETECTOR_HPP
#define SPHERE_DETECTOR_HPP

#include "processing/vision/object_detector.hpp"

namespace processing::vision {
    class SphereDetector : public ObjectDetector {
    public:
        // Detect the sphere enclosing the largest object in the image
        [[nodiscard]] std::pair<cv::Point2f, float> detect(const cv::Mat &image) const override;
    };
}

#endif // SPHERE_DETECTOR_HPP
