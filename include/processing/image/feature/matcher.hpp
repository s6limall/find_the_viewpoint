// File: processing/image/feature/matcher.hpp

#ifndef FEATURE_MATCHER_HPP
#define FEATURE_MATCHER_HPP

#include <opencv2/opencv.hpp>
#include <memory>

namespace processing::image {
    enum class MatcherType {
        FLANN,
        BF,
        SUPERGLUE,
    };

    class FeatureMatcher {
    public:
        virtual ~FeatureMatcher() = default;

        // Match features between two images and return the matches.
        [[nodiscard]] virtual std::vector<cv::DMatch> match(const cv::Mat &descriptors1,
                                                            const cv::Mat &descriptors2) const = 0;

        // Template-based factory method to create a Matcher object.
        template<typename MatcherType>
        static std::unique_ptr<FeatureMatcher> create() {
            return std::make_unique<MatcherType>();
        }
    };
}

#endif //FEATURE_MATCHER_HPP
