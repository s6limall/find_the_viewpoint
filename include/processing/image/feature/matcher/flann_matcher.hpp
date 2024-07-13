// File: processing/image/feature/matcher/flann_matcher.hpp

#ifndef FEATURE_MATCHER_FLANN_HPP
#define FEATURE_MATCHER_FLANN_HPP

#include "processing/image/feature/matcher.hpp"

namespace processing::image {
    class FLANNMatcher final : public FeatureMatcher {
    public:
        [[nodiscard]] std::vector<cv::DMatch>
        match(const cv::Mat &descriptors1, const cv::Mat &descriptors2) const override;

        void knnMatch(const cv::Mat &descriptors1, const cv::Mat &descriptors2,
                      std::vector<std::vector<cv::DMatch> > &knnMatches, int k) const override;

    private:
        static cv::Mat convertDescriptorsToFloat(const cv::Mat &descriptors);

        static std::vector<cv::DMatch> filterMatches(const std::vector<std::vector<cv::DMatch> > &knn_matches,
                                                     float ratio_thresh);


    };
}

#endif //FEATURE_MATCHER_FLANN_HPP
