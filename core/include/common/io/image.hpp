// File: common/io/image.hpp

#ifndef COMMON_IMAGE_IO_HPP
#define COMMON_IMAGE_IO_HPP

#include <filesystem>
#include <functional>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include "common/io/io.hpp"
#include "common/logging/logger.hpp"

namespace common::io::image {

    namespace fs = std::filesystem;

    /**
     * @brief Reads an image from a file using OpenCV.
     *
     * @param file_path Path to the image file.
     * @param mode Flag specifying the color type of the loaded image.
     * @return cv::Mat The loaded image.
     * @throws std::invalid_argument if the file path is empty or invalid.
     * @throws std::runtime_error if the image could not be read.
     */
    [[nodiscard]] inline cv::Mat readImage(const fs::path &file_path, const cv::ImreadModes mode = cv::IMREAD_COLOR) {
        if (file_path.empty()) {
            LOG_ERROR("Empty file path provided");
            throw std::invalid_argument("Empty file path provided");
        }

        if (!fs::exists(file_path)) {
            LOG_ERROR("File does not exist: {}", file_path.string());
            throw std::invalid_argument(fmt::format("File does not exist: {}", file_path.string()));
        }

        LOG_TRACE("Reading image from file: {}", file_path.string());
        cv::Mat image = cv::imread(file_path.string(), mode);
        if (image.empty()) {
            LOG_ERROR("Could not read image: {}", file_path.string());
            throw std::runtime_error(fmt::format("Could not read image: {}", file_path.string()));
        }
        LOG_TRACE("Image read successfully: {}", file_path.string());
        return image;
    }

    /**
     * @brief Writes an image to a file using OpenCV.
     *
     * @param file_path Path to the image file.
     * @param image Image to be saved.
     * @param create_directories Flag indicating whether to create directories if they do not exist.
     * @throws std::invalid_argument if the file path is empty or the image is empty.
     * @throws std::runtime_error if the image could not be written or directories could not be created.
     */
    inline void writeImage(const fs::path &file_path, const cv::Mat &image, const bool create_directories = true) {
        if (file_path.empty()) {
            LOG_ERROR("Empty file path provided");
            throw std::invalid_argument("Empty file path provided");
        }

        if (image.empty()) {
            LOG_ERROR("Attempt to write an empty image");
            throw std::invalid_argument("Attempt to write an empty image");
        }

        const fs::path parent_path = file_path.parent_path();
        if (create_directories && !parent_path.empty() && !fs::exists(parent_path)) {
            try {
                fs::create_directories(parent_path);
                LOG_DEBUG("Created directory: {}", parent_path.string());
            } catch (const fs::filesystem_error &e) {
                LOG_ERROR("Failed to create directory: {}. Error: {}", parent_path.string(), e.what());
                throw std::runtime_error(
                        fmt::format("Failed to create directory: {}. Error: {}", parent_path.string(), e.what()));
            }
        }

        LOG_TRACE("Writing image to file: {}", file_path.string());
        std::vector<int> params;
        std::string ext = file_path.extension().string();
        std::ranges::transform(ext, ext.begin(), ::tolower);

        if (ext == ".jpg" || ext == ".jpeg") {
            params.push_back(cv::IMWRITE_JPEG_QUALITY);
            params.push_back(95); // Set JPEG quality to 95%
        } else if (ext == ".png") {
            params.push_back(cv::IMWRITE_PNG_COMPRESSION);
            params.push_back(9); // Set PNG compression level to maximum
        }

        try {
            if (!cv::imwrite(file_path.string(), image, params)) {
                throw std::runtime_error("OpenCV imwrite failed");
            }
            LOG_TRACE("Image written successfully: {}", file_path.string());
        } catch (const cv::Exception &e) {
            LOG_ERROR("OpenCV exception while writing image: {}. Error: {}", file_path.string(), e.what());
            throw std::runtime_error(
                    fmt::format("Could not write image: {}. OpenCV Error: {}", file_path.string(), e.what()));
        } catch (const std::exception &e) {
            LOG_ERROR("Exception while writing image: {}. Error: {}", file_path.string(), e.what());
            throw std::runtime_error(fmt::format("Could not write image: {}. Error: {}", file_path.string(), e.what()));
        }
    }

    /**
     * @brief Resizes an image to a specified width while maintaining aspect ratio.
     *
     * @param image Input image.
     * @param target_width Desired width of the resized image.
     * @return cv::Mat Resized image.
     * @throws std::invalid_argument if the input image is empty or target width is non-positive.
     */
    [[nodiscard]] inline cv::Mat resizeImageToWidth(const cv::Mat &image, int target_width) {
        if (image.empty()) {
            LOG_ERROR("Attempt to resize an empty image");
            throw std::invalid_argument("Attempt to resize an empty image");
        }

        if (target_width <= 0) {
            LOG_ERROR("Invalid target width: {}", target_width);
            throw std::invalid_argument(fmt::format("Invalid target width: {}", target_width));
        }

        const double aspect_ratio = static_cast<double>(image.cols) / image.rows;
        const int target_height = static_cast<int>(target_width / aspect_ratio);

        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(target_width, target_height), 0, 0, cv::INTER_AREA);

        return resized_image;
    }

    /**
     * @brief Applies a gaussian blur to an image.
     *
     * @param image Input image.
     * @param kernel_size Size of the gaussian kernel. Must be odd and positive.
     * @return cv::Mat Blurred image.
     * @throws std::invalid_argument if the input image is empty or kernel size is invalid.
     */
    [[nodiscard]] inline cv::Mat applyGaussianBlur(const cv::Mat &image, int kernel_size) {
        if (image.empty()) {
            LOG_ERROR("Attempt to blur an empty image");
            throw std::invalid_argument("Attempt to blur an empty image");
        }

        if (kernel_size % 2 == 0 || kernel_size <= 0) {
            LOG_ERROR("Invalid kernel size: {}. Must be odd and positive.", kernel_size);
            throw std::invalid_argument(fmt::format("Invalid kernel size: {}. Must be odd and positive.", kernel_size));
        }

        cv::Mat blurred_image;
        cv::GaussianBlur(image, blurred_image, cv::Size(kernel_size, kernel_size), 0);

        return blurred_image;
    }
} // namespace common::io::image

#endif // COMMON_IMAGE_IO_HPP
