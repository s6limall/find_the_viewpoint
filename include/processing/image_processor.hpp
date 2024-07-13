// File: processing/image_processor.hpp

#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

#include <optional>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <vector>

#include "common/logging/logger.hpp"

namespace processing::image {
    class ImageProcessor {
    public:
        // Compare two images for similarity using SIFT feature matching
        // Static method to compare two images, returning a boolean match and a similarity score.
        [[nodiscard]] static std::pair<bool, double> compareImages(const cv::Mat &image1, const cv::Mat &image2);

    };


}

#endif // IMAGE_PROCESSOR_HPP
