// File: processing/image/feature/extractor/sift.hpp

#ifndef FEATURE_EXTRACTOR_SIFT_HPP
#define FEATURE_EXTRACTOR_SIFT_HPP

#include "processing/image/feature/extractor.hpp"

namespace processing::image {
    class SIFTExtractor final : public FeatureExtractor {
    public:
        SIFTExtractor();

        [[nodiscard]] std::pair<KeyPoints, Descriptors> extract(const cv::Mat &image) const override;

    private:
        cv::Ptr<cv::SIFT> sift_;

    };
}

#endif //FEATURE_EXTRACTOR_SIFT_HPP
