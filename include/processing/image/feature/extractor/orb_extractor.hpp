// File: processing/image/feature/extractor/orb.hpp

#ifndef FEATURE_EXTRACTOR_ORB_HPP
#define FEATURE_EXTRACTOR_ORB_HPP

#include "processing/image/feature/extractor.hpp"

namespace processing::image {
    class ORBExtractor : public FeatureExtractor {
    public:
        [[nodiscard]] std::pair<std::vector<cv::KeyPoint>, cv::Mat> extract(const cv::Mat& image) const override;
    };
}

#endif //FEATURE_EXTRACTOR_ORB_HPP
