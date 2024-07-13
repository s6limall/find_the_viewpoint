// File: processing/image/feature/extractor/sift.cpp

#include "processing/image/feature/extractor/sift_extractor.hpp"

#include "config/configuration.hpp"

namespace processing::image {

    SIFTExtractor::SIFTExtractor() {
        sift_ = cv::SIFT::create();
        if (sift_.empty()) {
            LOG_ERROR("Failed to create SIFT feature extractor.");
            throw std::runtime_error("Failed to create SIFT feature extractor.");
        }
    }

    std::pair<KeyPoints, Descriptors> SIFTExtractor::extract(const cv::Mat &image) const {
        if (image.empty()) {
            throw std::invalid_argument("Input image is empty.");
        }

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        sift_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        return {std::move(keypoints), std::move(descriptors)};

    }
}
