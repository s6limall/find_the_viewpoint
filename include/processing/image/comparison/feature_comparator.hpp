// File: processing/image/comparison/feature_comparator.hpp

#ifndef IMAGE_COMPARATOR_FEATURE_MATCHING_HPP
#define IMAGE_COMPARATOR_FEATURE_MATCHING_HPP

#include "processing/image/comparator.hpp"
#include "processing/image/feature/extractor.hpp"
#include "processing/image/feature/matcher.hpp"

namespace processing::image {
    class FeatureComparator : public ImageComparator {
    public:
        [[nodiscard]] double compare(const cv::Mat &image1, const cv::Mat &image2) const override;
    };
}

#endif //IMAGE_COMPARATOR_FEATURE_MATCHING_HPP
