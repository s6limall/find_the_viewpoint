// File: processing/image/feature/extractor.hpp

#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace processing::image {
    enum class ExtractorType {
        SIFT,
        ORB
    };

    class FeatureExtractor {
    public:
        virtual ~FeatureExtractor() = default;

        // Extract features from an image and return the keypoints and descriptors.
        [[nodiscard]] virtual std::pair<std::vector<cv::KeyPoint>, cv::Mat> extract(const cv::Mat &image) const = 0;

        // Template-based static factory method to create an extractor object.
        template<typename ExtractorType>
        static std::unique_ptr<FeatureExtractor> create() {
            return std::make_unique<ExtractorType>();
        }
    };
}

#endif //FEATURE_EXTRACTOR_HPP
