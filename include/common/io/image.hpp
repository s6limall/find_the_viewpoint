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
     * @param file_path Path to the image file as a string or string_view.
     * @param mode Flag specifying the color type of the loaded image.
     * @return cv::Mat The loaded image.
     * @throws std::runtime_error if the image could not be read.
     */
    inline cv::Mat readImage(std::string_view file_path, const cv::ImreadModes mode = cv::IMREAD_COLOR) {
        // Create parent directories if they do not exist
        if (const std::filesystem::path path(file_path); !path.parent_path().empty() && !exists(path.parent_path())) {
            LOG_DEBUG("filepath = {}", file_path);
            createDirectory(path.parent_path().string());
        }

        LOG_TRACE("Reading image from file: {}", file_path);
        cv::Mat image = cv::imread(std::string(file_path), mode);
        if (image.empty()) {
            LOG_ERROR("Could not read image: {}", file_path);
            throw std::runtime_error(fmt::format("Could not read image: {}", file_path));
        }
        LOG_TRACE("Image read successfully: {}", file_path);
        return image;
    }

    /**
     * @brief Writes an image to a file using OpenCV.
     *
     * @param file_path Path to the image file as a string or string_view.
     * @param image Image to be saved.
     * @param create_directories Flag indicating whether to create directories if they do not exist.
     * @throws std::runtime_error if the image could not be written.
     */
    inline void writeImage(std::string_view file_path, const cv::Mat &image, const bool create_directories = true) {
        if (const std::filesystem::path path(file_path); create_directories) {
            common::io::createDirectory(path.parent_path().string());
        }

        LOG_TRACE("Writing image to file: {}", file_path);
        if (!cv::imwrite(std::string(file_path), image)) {
            LOG_ERROR("Could not write image: {}", file_path);
            throw std::runtime_error(fmt::format("Could not write image: {}", file_path));
        }
        LOG_TRACE("Image written successfully: {}", file_path);
    }
}

#endif // COMMON_IMAGE_IO_HPP
