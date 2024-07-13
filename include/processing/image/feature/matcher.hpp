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

        // K-Nearest Neighbors matching
        // KNN match features between two images and return the matches.
        virtual void knnMatch(const cv::Mat &descriptors1,
                              const cv::Mat &descriptors2,
                              std::vector<std::vector<cv::DMatch> > &knnMatches,
                              int k) const = 0;

        template<typename T>
        static std::unique_ptr<FeatureMatcher> create() {
            return std::make_unique<T>();
        }
    };
}

#endif //FEATURE_MATCHER_HPP
