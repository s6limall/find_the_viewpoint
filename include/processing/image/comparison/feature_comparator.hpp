// File: processing/image/feature_comparator.hpp

#ifndef FEATURE_COMPARATOR_HPP
#define FEATURE_COMPARATOR_HPP

#include "processing/image/comparator.hpp"
#include "processing/image/feature/extractor.hpp"
#include "processing/image/feature/matcher.hpp"
#include "types/image.hpp"

namespace processing::image {

    class FeatureComparator final : public ImageComparator {
    public:
        FeatureComparator(std::unique_ptr<FeatureExtractor> extractor, std::unique_ptr<FeatureMatcher> matcher);

        [[nodiscard]] double compare(const cv::Mat &image1, const cv::Mat &image2) const override;
        [[nodiscard]] double compare(const Image<> &img1, const Image<> &img2) const;

    private:
        std::unique_ptr<FeatureExtractor> extractor_;
        std::unique_ptr<FeatureMatcher> matcher_;

        [[nodiscard]] double compareDescriptors(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                                                size_t keypoints1, size_t keypoints2) const;
    };
} // namespace processing::image

#endif // FEATURE_COMPARATOR_HPP
