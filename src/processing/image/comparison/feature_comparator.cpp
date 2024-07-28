// File: processing/image/comparison/feature_comparator.cpp

#include "processing/image/comparison/feature_comparator.hpp"

namespace processing::image {

    FeatureComparator::FeatureComparator(std::shared_ptr<FeatureExtractor> extractor,
                                         std::shared_ptr<FeatureMatcher> matcher) :
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
            return 0.0;
        }

        // Match descriptors
        const auto matches = matcher_->match(descriptors1, descriptors2);

        if (matches.empty()) {
            LOG_WARN("No matches found.");
            return 0.0;
        }

        // Calculate the match quality score
        const double max_matches = std::min(keypoints1, keypoints2);
        double score = static_cast<double>(matches.size()) / max_matches;

        // Clamp score to [0, 1]
        score = std::clamp(score, 0.0, 1.0);

        return score;
    }
} // namespace processing::image
