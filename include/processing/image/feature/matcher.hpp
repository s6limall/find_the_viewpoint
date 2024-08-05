// File: processing/image/feature/matcher.hpp

#ifndef FEATURE_MATCHER_HPP
#define FEATURE_MATCHER_HPP

#include <memory>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>
#include "common/logging/logger.hpp"

namespace processing::image {

    class FeatureMatcher {
    public:
        virtual ~FeatureMatcher() = default;

        using Features = std::pair<cv::Mat, std::vector<cv::KeyPoint>>;
        using MatchResult = std::tuple<std::vector<cv::DMatch>, cv::Mat, std::vector<uchar>>;

        [[nodiscard]] virtual MatchResult match(const Features &features1, const Features &features2) const = 0;

        template<typename T>
        static std::shared_ptr<FeatureMatcher> create() {
            static_assert(std::is_base_of_v<FeatureMatcher, T>, "T must derive from FeatureMatcher");
            return std::make_shared<T>();
        }
    };

} // namespace processing::image

#endif // FEATURE_MATCHER_HPP
