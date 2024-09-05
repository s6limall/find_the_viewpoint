// File: common/utilities/image.hpp

#ifndef COMMON_UTILITIES_IMAGE_HPP
#define COMMON_UTILITIES_IMAGE_HPP

#include <opencv2/img_hash.hpp>
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
        if (image.channels() != 3) {
            LOG_TRACE("Image is already grayscale");
            return image;
        }
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
     * @brief Computes the standard deviation of the grayscale version of the image.
     *
     * @param image Input image.
     * @return double Standard deviation of the grayscale image.
     */
    inline double computeStandardDeviation(const cv::Mat &image) noexcept {
        cv::Scalar stddev;
        cv::meanStdDev(toGrayscale(image), cv::noArray(), stddev);
        return stddev[0];
    }

    /**
     * @brief Displays an image in a window.
     *
     * @param window_name Name of the window.
     * @param image Image to be displayed.
     */
    inline void display(const cv::Mat &image, const std::string &window_name = "Image") {
        cv::Mat display;
        if (image.channels() == 1) {
            // Normalize for visualization
            cv::normalize(image, display, 0, 255, cv::NORM_MINMAX, CV_8U);
        } else {
            display = image.clone();
            display.convertTo(display, CV_8U, 255.0);
        }
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        cv::imshow(window_name, image);
        cv::waitKey(0);
        cv::destroyWindow(window_name);
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


    inline auto toPyramid(const cv::Mat &image, const int levels) noexcept -> std::vector<cv::Mat> {
        std::vector<cv::Mat> pyramid;
        pyramid.reserve(levels);
        pyramid.push_back(image); // Add the original image at the first level

        cv::Mat currentImage = image;
        for (int i = 1; i < levels; ++i) {
            cv::Mat down;
            cv::pyrDown(currentImage, down);
            pyramid.push_back(down);
            currentImage = down;
        }

        return pyramid;
    }

    /**
     * @brief Computes the perceptual hash of an image using the average hashing method.
     *
     * @param image Input image.
     * @return cv::Mat The perceptual hash.
     */
    inline cv::Mat computePerceptualHash(const cv::Mat &image) {
        cv::Mat hash;
        cv::img_hash::averageHash(image, hash);
        return hash;
    }

} // namespace common::utilities

#endif // COMMON_UTILITIES_IMAGE_HPP
