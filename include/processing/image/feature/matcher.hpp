// File: processing/image/feature/matcher.hpp

#ifndef FEATURE_MATCHER_HPP
#define FEATURE_MATCHER_HPP

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

namespace processing::image {
    enum class MatcherType {
        FLANN,
        BF,
        SUPERGLUE,
    };

    class FeatureMatcher {
    public:
        virtual ~FeatureMatcher() = default;

        [[nodiscard]] virtual std::vector<cv::DMatch> match(const cv::Mat &descriptors1,
                                                            const cv::Mat &descriptors2) const = 0;

        template<typename T>
        static std::unique_ptr<FeatureMatcher> create() {
            static_assert(std::is_base_of_v<FeatureMatcher, T>, "T must derive from FeatureMatcher");
            return std::make_unique<T>();
        }
    };
}

#endif //FEATURE_MATCHER_HPP
