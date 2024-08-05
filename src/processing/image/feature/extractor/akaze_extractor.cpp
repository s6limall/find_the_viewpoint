/*
// File: processing/image/feature/extractor/akaze_extractor.cpp

#include "processing/image/feature/extractor/akaze_extractor.hpp"
#include "common/logging/logger.hpp"

namespace processing::image {

    AKAZEExtractor::AKAZEExtractor() {
        akaze_ = cv::AKAZE::create();
        if (akaze_.empty()) {
            LOG_ERROR("Failed to create AKAZE feature extractor.");
            throw std::runtime_error("Failed to create AKAZE feature extractor.");
        }
    }

    std::pair<KeyPoints, Descriptors> AKAZEExtractor::extract(const cv::Mat &image) const noexcept {
        KeyPoints keypoints;
        Descriptors descriptors;

        akaze_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        return std::make_pair(std::move(keypoints), std::move(descriptors));
    }

} // namespace processing::image
*/
