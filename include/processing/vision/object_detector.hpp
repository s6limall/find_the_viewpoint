// File: processing/vision/object_detector.hpp

#ifndef OBJECT_DETECTOR_HPP
#define OBJECT_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <utility>

namespace processing::vision {
    class ObjectDetector {
    public:
        virtual ~ObjectDetector() = default;

        /**
         * @brief Detects the object in the image and returns its center and radius.
         *
         * @param image The input image.
         * @return A pair consisting of the center and radius of the detected object.
         */
        virtual std::pair<cv::Point2f, float> detect(const cv::Mat &image) const = 0;
    };
}

#endif // OBJECT_DETECTOR_HPP

