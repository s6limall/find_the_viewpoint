// File: processing/image_processor.cpp

#include "processing/image_processor.hpp"
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "common/logging/logger.hpp"
#include "config/configuration.hpp"
#include "processing/image/comparison/feature_comparator.hpp"
#include "processing/image/comparison/mse_comparator.hpp"
#include "processing/image/comparison/ssim_comparator.hpp"
#include "processing/image/feature/extractor/akaze_extractor.hpp"
#include "processing/image/feature/matcher/flann_matcher.hpp"

namespace processing::image {

    std::pair<bool, double> ImageProcessor::compareImages(const cv::Mat &image1, const cv::Mat &image2) {
        if (image1.empty() || image2.empty()) {
            throw std::invalid_argument("One or both images are empty");
        }

        const std::string &comparatorType = config::get("image_comparator.type", "SSIM");

        LOG_TRACE("Using image comparator: {}", comparatorType);

        // Factory map for creating the appropriate comparator
        static const std::map<std::string, std::function<std::unique_ptr<ImageComparator>()>> comparatorFactory = {
                {"MSE", []() { return std::make_unique<MSEComparator>(); }},
                {"SSIM", []() { return std::make_unique<SSIMComparator>(); }},
                {"FeatureBased", []() {
                     auto extractor = std::make_unique<AKAZEExtractor>();
                     auto matcher = std::make_unique<FLANNMatcher>();
                     return std::make_unique<FeatureComparator>(std::move(extractor), std::move(matcher));
                 }}};

        auto it = comparatorFactory.find(comparatorType);
        if (it == comparatorFactory.end()) {
            throw std::runtime_error("Unsupported image comparator type: " + comparatorType);
        }

        const std::unique_ptr<ImageComparator> comparator = it->second();

        double similarity = comparator->compare(image1, image2);
        // LOG_DEBUG("Image similarity: {}", similarity);

        // Use structured bindings to retrieve thresholds in a concise manner
        const std::map<std::string, double> thresholds = {
                {"MSE", config::get("image_comparator.mse.threshold", 1000.0)},
                {"SSIM", config::get("image_comparator.ssim.threshold", 0.5)},
                {"FeatureBased", config::get("image_comparator.feature_based.threshold", 0.5)}};

        const double threshold = thresholds.at(comparatorType);

        LOG_TRACE("Threshold: {}", threshold);

        // Determine if the images match based on the comparator type and threshold
        const bool isMatch = (comparatorType == "SSIM" || comparatorType == "FeatureBased") ? (similarity >= threshold)
                                                                                            : (similarity <= threshold);

        LOG_DEBUG("Returning with isMatch: {}, similarity: {}", isMatch, similarity);

        return std::make_pair(isMatch, similarity);
    }
} // namespace processing::image
