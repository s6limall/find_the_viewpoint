// File: processing/image/comparison/feature_comparator.hpp

#ifndef IMAGE_COMPARATOR_FEATURE_MATCHING_HPP
#define IMAGE_COMPARATOR_FEATURE_MATCHING_HPP

#include <numeric>
#include <stdexcept>
#include <map>
#include <memory>
#include <future>

#include "processing/image/comparator.hpp"
#include "processing/image/feature/extractor.hpp"
#include "processing/image/feature/matcher.hpp"
#include "processing/image/feature/extractor/sift_extractor.hpp"
#include "processing/image/feature/extractor/orb_extractor.hpp"
#include "processing/image/feature/matcher/flann_matcher.hpp"
#include "processing/image/feature/matcher/bf_matcher.hpp"

namespace processing::image {
    class FeatureComparator final : public ImageComparator {
    public:
        ~FeatureComparator() override = default;

        [[nodiscard]] double compare(const cv::Mat &image1, const cv::Mat &image2) const override;
    };
}

#endif //IMAGE_COMPARATOR_FEATURE_MATCHING_HPP
