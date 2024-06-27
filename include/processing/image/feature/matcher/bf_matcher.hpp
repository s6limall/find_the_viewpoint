// File: processing/image/feature/matcher/bf.hpp

#ifndef FEATURE_MATCHER_BF_HPP
#define FEATURE_MATCHER_BF_HPP

#include "processing/image/feature/matcher.hpp"

namespace processing::image {
    class BFMatcher: public FeatureMatcher {
    public:
        std::vector<cv::DMatch> match(const cv::Mat& descriptors1, const cv::Mat& descriptors2) const override;
    };
}

#endif //FEATURE_MATCHER_BF_HPP
