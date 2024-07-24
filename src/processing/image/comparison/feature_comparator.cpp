// File: processing/image/feature_comparator.cpp

#include "processing/image/comparison/feature_comparator.hpp"
#include <algorithm> // for std::clamp
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <stdexcept>

namespace processing::image {

    FeatureComparator::FeatureComparator(std::unique_ptr<FeatureExtractor> extractor,
                                         std::unique_ptr<FeatureMatcher> matcher) :
        extractor_(std::move(extractor)), matcher_(std::move(matcher)) {}

    double FeatureComparator::compare(const cv::Mat &image1, const cv::Mat &image2) const {
        // Extract keypoints and descriptors for both images
        auto [keypoints1, descriptors1] = extractor_->extract(image1);
        auto [keypoints2, descriptors2] = extractor_->extract(image2);

        // Compare the descriptors
        return compareDescriptors(descriptors1, descriptors2, keypoints1.size(), keypoints2.size());
    }

    double FeatureComparator::compare(const Image<> &img1, const Image<> &img2) const {
        // Compare the precomputed descriptors and keypoints
        return compareDescriptors(img1.getDescriptors(), img2.getDescriptors(), img1.getKeypoints().size(),
                                  img2.getKeypoints().size());
    }

    double FeatureComparator::compareDescriptors(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                                                 const size_t keypoints1, const size_t keypoints2) const {
        // Check if descriptors are empty
        if (descriptors1.empty() || descriptors2.empty()) {
            LOG_WARN("Descriptors are empty for one or both images");
            // throw std::invalid_argument("Descriptors are empty for one or both images");
        }

        // Match descriptors
        const auto matches = matcher_->match(descriptors1, descriptors2);

        // Calculate total number of keypoints
        const size_t totalKeypoints = keypoints1 + keypoints2;

        // Avoid division by zero
        if (totalKeypoints == 0) {
            return 0.0;
        }

        // Calculate match ratio
        const double matchRatio = static_cast<double>(matches.size()) / static_cast<double>(totalKeypoints);

        // Normalize match ratio to range [0, 1]
        return std::clamp(matchRatio * 2.0, 0.0, 1.0);
    }
} // namespace processing::image
