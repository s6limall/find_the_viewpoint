// File: processing/image/feature/matcher/flann_matcher.hpp

#ifndef FEATURE_MATCHER_FLANN_HPP
#define FEATURE_MATCHER_FLANN_HPP

#include <exception>
#include <opencv2/opencv.hpp>
#include "config/configuration.hpp"
#include "processing/image/feature/matcher.hpp"

namespace processing::image {
    class FLANNMatcher final : public FeatureMatcher {
    public:
        [[nodiscard]] std::vector<cv::DMatch> match(const cv::Mat &descriptors1,
                                                    const cv::Mat &descriptors2) const override;

    private:
        [[nodiscard]] static cv::Mat convertDescriptorsToFloat(const cv::Mat &descriptors);

        [[nodiscard]] static std::vector<cv::DMatch>
        filterMatches(const std::vector<std::vector<cv::DMatch>> &knnMatches, float ratioThresh);

        [[nodiscard]] static std::vector<std::vector<cv::DMatch>> knnMatch(const cv::Mat &desc1, const cv::Mat &desc2,
                                                                           int k);
    };
} // namespace processing::image

#endif // FEATURE_MATCHER_FLANN_HPP
