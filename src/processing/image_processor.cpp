// File: processing/image_processor.cpp

#include "processing/image_processor.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

#include "common/logging/logger.hpp"
#include "config/configuration.hpp"
#include "processing/image/comparison/feature_comparator.hpp"
#include "processing/image/comparison/mse_comparator.hpp"
#include "processing/image/comparison/ssim_comparator.hpp"

namespace processing::image {
    // Convert image to grayscale
    cv::Mat convertToGrayscale(const cv::Mat &image) {
        spdlog::debug("Converting image to grayscale.");
        if (image.empty()) {
            spdlog::error("Input image is empty.");
            return {};
        }
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY); // Convert BGR image to grayscale
        spdlog::debug("Image converted to grayscale.");
        return gray;
    }

    std::pair<bool, double> ImageProcessor::compareImages(const cv::Mat &image1, const cv::Mat &image2) {
        if (image1.empty() || image2.empty()) {
            throw std::invalid_argument("One or both images are empty");
        }

        const auto &config = config::Configuration::getInstance();
        const std::string &comparatorType = config.get<std::string>("image_comparator.type", "SSIM");

        spdlog::info("Using image comparator: {}", comparatorType);

        // Factory map for creating the appropriate comparator
        static const std::map<std::string, std::function<std::unique_ptr<ImageComparator>()> > comparatorFactory = {
            {"MSE", []() { return std::make_unique<MSEComparator>(); }},
            {"SSIM", []() { return std::make_unique<SSIMComparator>(); }},
            {"FeatureBased", []() { return std::make_unique<FeatureComparator>(); }}
        };

        auto it = comparatorFactory.find(comparatorType);
        if (it == comparatorFactory.end()) {
            throw std::runtime_error("Unsupported image comparator type: " + comparatorType);
        }

        std::unique_ptr<ImageComparator> comparator = it->second();

        double similarity = comparator->compare(image1, image2);
        spdlog::debug("Image similarity: {}", similarity);

        // Use structured bindings to retrieve thresholds in a concise manner
        const auto thresholds = std::map<std::string, double>{
            {"MSE", config.get<double>("image_comparator.mse.threshold", 1000.0)},
            {"SSIM", config.get<double>("image_comparator.ssim.threshold", 0.5)},
            {"FeatureBased", config.get<double>("image_comparator.feature_based.threshold", 50.0)}
        };

        const auto threshold = thresholds.at(comparatorType);

        spdlog::debug("Threshold: {}", threshold);

        // Determine if the images match based on the comparator type and threshold
        const bool isMatch = (comparatorType == "SSIM") ? (similarity >= threshold) : (similarity <= threshold);

        spdlog::debug("Returning with isMatch: {}, similarity: {}", isMatch, similarity);

        return std::make_pair(isMatch, similarity);
    }
}
