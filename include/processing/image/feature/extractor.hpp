// File: processing/image/feature/extractor.hpp

#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>


namespace processing::image {

    using KeyPoints = std::vector<cv::KeyPoint>;
    using Descriptors = cv::Mat;

    enum class ExtractorType {
        SIFT,
        ORB,
        SUPERPOINT,
        AKAZE,
    };

    class FeatureExtractor {
    public:
        virtual ~FeatureExtractor() = default;

        // Extract features from an image and return the keypoints and descriptors.
        [[nodiscard]] virtual std::pair<KeyPoints, Descriptors> extract(const cv::Mat &image) const = 0;

        // Static factory method to create an extractor object.
        template<typename T>
        static std::unique_ptr<FeatureExtractor> create() {
            return std::make_unique<T>();
        }
    };
} // namespace processing::image

#endif // FEATURE_EXTRACTOR_HPP