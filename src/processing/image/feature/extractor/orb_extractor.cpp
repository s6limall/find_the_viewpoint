// File: processing/image/feature/extractor/orb.cpp

#include "processing/image/feature/extractor/orb_extractor.hpp"

#include "config/configuration.hpp"

namespace processing::image {

    ORBExtractor::ORBExtractor() {
        orb_ = cv::ORB::create();
        if (orb_.empty()) {
            LOG_ERROR("Failed to create ORB feature extractor.");
            throw std::runtime_error("Failed to create ORB feature extractor.");
        }
    }

    std::pair<KeyPoints, Descriptors> ORBExtractor::extract(const cv::Mat &image) const {
        if (image.empty()) {
            throw std::invalid_argument("Input image is empty.");
        }

        KeyPoints keypoints;
        Descriptors descriptors;

        orb_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

        return {std::move(keypoints), std::move(descriptors)};
    }

}
