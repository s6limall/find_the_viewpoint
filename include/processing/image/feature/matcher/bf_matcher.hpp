// File: processing/image/feature/matcher/bf.hpp

#ifndef FEATURE_MATCHER_BF_HPP
#define FEATURE_MATCHER_BF_HPP

#include "processing/image/feature/matcher.hpp"

namespace processing::image {
    class BFMatcher final : public FeatureMatcher {
    public:
        [[nodiscard]] std::vector<cv::DMatch>
        match(const cv::Mat &descriptors1, const cv::Mat &descriptors2) const override;

        void knnMatch(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                      std::vector<std::vector<cv::DMatch> > &knnMatches, int k) const;
    };
}

#endif //FEATURE_MATCHER_BF_HPP
