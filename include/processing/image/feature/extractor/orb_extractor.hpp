// File: processing/image/feature/extractor/orb.hpp

#ifndef FEATURE_EXTRACTOR_ORB_HPP
#define FEATURE_EXTRACTOR_ORB_HPP

#include "processing/image/feature/extractor.hpp"

namespace processing::image {

    class ORBExtractor final : public FeatureExtractor {
    public:
        ORBExtractor();

        // Returns the keypoints and descriptors of the input image. {keypoints, descriptors}
        [[nodiscard]] std::pair<KeyPoints, Descriptors> extract(const cv::Mat &image) const override;

    private:
        cv::Ptr<cv::ORB> orb_;
    };
}

#endif //FEATURE_EXTRACTOR_ORB_HPP
