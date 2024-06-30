// File: common/io/image.hpp

#ifndef COMMON_IMAGE_IO_HPP
#define COMMON_IMAGE_IO_HPP

#include <opencv2/opencv.hpp>
#include <functional>
#include <filesystem>
#include "common/io/io.hpp"
#include "common/logging/logger.hpp"

namespace common::io::image {
    /**
     * @brief Reads an image from a file using OpenCV.
     *
     * @param file_path Path to the image file as a string.
     * @param mode Flag specifying the color type of the loaded image.
     * @return cv::Mat The loaded image.
     * @throws std::runtime_error if the image could not be read.
     */
    inline cv::Mat readImage(const std::string &file_path, const cv::ImreadModes mode = cv::IMREAD_COLOR) {
        std::filesystem::path path(file_path);

        // Create parent directories if they do not exist
        if (!std::filesystem::exists(path.parent_path())) {
            common::io::createDirectory(path.parent_path().string());
        }

        LOG_DEBUG("Reading image from file: {}", file_path);
        cv::Mat image = cv::imread(file_path, mode);
        if (image.empty()) {
            LOG_ERROR("Could not read image: {}", file_path);
            throw std::runtime_error(fmt::format("Could not read image: {}", file_path));
        }
        LOG_DEBUG("Image read successfully: {}", file_path);
        return image;
    }

    /**
     * @brief Writes an image to a file using OpenCV.
     *
     * @param file_path Path to the image file as a string.
     * @param image Image to be saved.
     * @param create_directories Flag indicating whether to create directories if they do not exist.
     * @throws std::runtime_error if the image could not be written.
     */
    inline void writeImage(const std::string &file_path, const cv::Mat &image,
                           const bool create_directories = true) {
        std::filesystem::path path(file_path);

        if (create_directories) {
            common::io::createDirectory(path.parent_path().string());
        }

        LOG_DEBUG("Writing image to file: {}", file_path);
        if (!cv::imwrite(file_path, image)) {
            LOG_ERROR("Could not write image: {}", file_path);
            throw std::runtime_error(fmt::format("Could not write image: {}", file_path));
        }
        LOG_DEBUG("Image written successfully: {}", file_path);
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

        LOG_DEBUG("Applying operation to each pixel of the image");
        cv::Mat result = image.clone();
        std::transform(image.begin<cv::Vec3b>(), image.end<cv::Vec3b>(), result.begin<cv::Vec3b>(), operation);
        LOG_DEBUG("Operation applied to all pixels successfully");
        return result;
    }
}

#endif // COMMON_IMAGE_IO_HPP
