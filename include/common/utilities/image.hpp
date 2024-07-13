// File: common/utilities/image.hpp

#ifndef COMMON_UTILITIES_IMAGE_HPP
#define COMMON_UTILITIES_IMAGE_HPP

#include <opencv2/opencv.hpp>
#include "common/logging/logger.hpp"

namespace common::utilities {

    /**
     * @brief Converts an image to grayscale.
     *
     * @param image Image to be converted.
     * @return cv::Mat The grayscale image.
     */
    inline cv::Mat toGrayscale(const cv::Mat &image) {
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        LOG_TRACE("Converted image to grayscale");
        return gray;
    }

    /**
     * @brief Convert an image to binary format using a threshold.
     *
     * If the input image is color, it converts it to grayscale first.
     *
     * @param image Input image (CV_8UC1 or CV_8UC3).
     * @param threshold Threshold value for binary conversion.
     * @param max_value Maximum value for binary conversion.
     * @return Binary image (CV_8UC1) where pixel values are either 0 or 255.
     *
     * @throws std::invalid_argument if the input image is empty.
     */
    inline cv::Mat toBinary(const cv::Mat &image, const double threshold = 128.0, const double max_value = 255.0) {
        // Check if the input image is empty
        if (image.empty()) {
            throw std::invalid_argument("Input image is empty.");
        }

        cv::Mat binaryImage;
        cv::threshold(toGrayscale(image), binaryImage, threshold, max_value, cv::THRESH_BINARY);

        return binaryImage;
    }

    /**
     * @brief Applies a generic operation to each pixel of an image using a lambda function.
     *
     * @param image Image to be processed.
     * @param operation Lambda function defining the operation to apply.
     * @return cv::Mat The processed image.
     * @throws std::runtime_error if the image is empty.
     */
    inline cv::Mat applyOperation(const cv::Mat &image, const std::function<cv::Vec3b(cv::Vec3b)> &operation) {
        if (image.empty()) {
            LOG_ERROR("Image is empty, cannot apply operation");
            throw std::runtime_error("Image is empty, cannot apply operation");
        }

        LOG_TRACE("Applying operation to each pixel of the image");
        cv::Mat result = image.clone();
        std::transform(image.begin<cv::Vec3b>(), image.end<cv::Vec3b>(), result.begin<cv::Vec3b>(), operation);
        LOG_TRACE("Operation applied to all pixels successfully");
        return result;
    }

}

#endif //COMMON_UTILITIES_IMAGE_HPP
