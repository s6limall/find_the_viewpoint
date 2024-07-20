// File: processing/image/feature/extractor/akaze_extractor.hpp

#ifndef AKAZE_EXTRACTOR_HPP
#define AKAZE_EXTRACTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "processing/image/feature/extractor.hpp"

namespace processing::image {

    class AKAZEExtractor final : public FeatureExtractor {
    public:
        AKAZEExtractor();

        [[nodiscard]] std::pair<KeyPoints, Descriptors> extract(const cv::Mat &image) const noexcept override;

    private:
        cv::Ptr<cv::AKAZE> akaze_;
    };


}
#endif // AKAZE_EXTRACTOR_HPP
