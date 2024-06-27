// File: processing/image/comparison/feature_comparator.cpp

#include "processing/image/comparison/feature_comparator.hpp"
#include "processing/image/feature/extractor/sift_extractor.hpp"
#include "processing/image/feature/extractor/orb_extractor.hpp"
#include "processing/image/feature/matcher/bf_matcher.hpp"
#include "processing/image/feature/matcher/flann_matcher.hpp"
#include <numeric>
#include <stdexcept>
#include <map>
#include <memory>

#include "config/configuration.hpp"

namespace processing::image {
    double FeatureComparator::compare(const cv::Mat &image1, const cv::Mat &image2) const {
        // Check if images are empty
        if (image1.empty() || image2.empty()) {
            spdlog::error("One or both images are empty");
            throw std::invalid_argument("One or both images are empty");
        }

        // Log configuration parameters
        const auto &config = config::Configuration::getInstance();
        const auto extractorType = config.get<std::string>("feature_detector.type", "SIFT");
        const auto matcherType = config.get<std::string>("feature_matcher.type", "FLANN");
        spdlog::debug("Feature Comparator Configuration - Extractor Type: {}, Matcher Type: {}", extractorType,
                      matcherType);

        // Map to store factory functions for extractors
        using ExtractorFactory = std::function<std::unique_ptr<FeatureExtractor>()>;
        static const std::map<std::string, ExtractorFactory> extractorFactoryMap = {
            {"SIFT", []() { return FeatureExtractor::create<SIFTExtractor>(); }},
            {"ORB", []() { return FeatureExtractor::create<ORBExtractor>(); }}
        };

        // Map to store factory functions for matchers
        using MatcherFactory = std::function<std::unique_ptr<FeatureMatcher>()>;
        static const std::map<std::string, MatcherFactory> matcherFactoryMap = {
            {"FLANN", []() { return FeatureMatcher::create<FLANNMatcher>(); }},
            {"BF", []() { return FeatureMatcher::create<BFMatcher>(); }}
        };

        // Create extractor
        auto extractorIt = extractorFactoryMap.find(extractorType);
        if (extractorIt == extractorFactoryMap.end()) {
            spdlog::error("Unsupported feature extractor type: {}", extractorType);
            throw std::runtime_error("Unsupported feature extractor type: " + extractorType);
        }
        std::unique_ptr<FeatureExtractor> extractor = extractorIt->second();
        spdlog::debug("Using {} feature extractor", extractorType);

        // Create matcher
        auto matcherIt = matcherFactoryMap.find(matcherType);
        if (matcherIt == matcherFactoryMap.end()) {
            spdlog::error("Unsupported feature matcher type: {}", matcherType);
            throw std::runtime_error("Unsupported feature matcher type: " + matcherType);
        }
        std::unique_ptr<FeatureMatcher> matcher = matcherIt->second();
        spdlog::debug("Using {} feature matcher", matcherType);

        // Extract keypoints and descriptors
        auto [keypoints1, descriptors1] = extractor->extract(image1);
        auto [keypoints2, descriptors2] = extractor->extract(image2);
        spdlog::debug("Extracted {} keypoints from image 1 and {} keypoints from image 2",
                      keypoints1.size(), keypoints2.size());
        spdlog::debug("Extracted {} descriptors from image 1 and {} descriptors from image 2",
                      descriptors1.rows, descriptors2.rows);

        // Perform matching
        std::vector<cv::DMatch> matches = matcher->match(descriptors1, descriptors2);

        // Calculate score based on matches
        const double score = std::accumulate(matches.begin(), matches.end(), 0.0,
                                             [](const double sum, const cv::DMatch &match) {
                                                 // Use inverse distance to avoid division by zero
                                                 return sum + 1.0 / (match.distance + 1e-5);
                                             });

        spdlog::info("Feature comparison completed with a score of {}", score);

        return score;
    }
}
