// File: processing/image/feature/extractor/sift.hpp

#ifndef FEATURE_EXTRACTOR_SIFT_HPP
#define FEATURE_EXTRACTOR_SIFT_HPP

#include "processing/image/feature/extractor.hpp"

namespace processing::image {
    class SIFTExtractor : public FeatureExtractor {
    public:
        std::pair<std::vector<cv::KeyPoint>, cv::Mat> extract(const cv::Mat& image) const override;
    };
}

#endif //FEATURE_EXTRACTOR_SIFT_HPP
