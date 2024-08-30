// File: processing/image/feature/extractor.cpp

#include "processing/image/feature/extractor.hpp"
#include "common/logging/logger.hpp"
#include "config/configuration.hpp"
#include "processing/image/feature/extractor/akaze_extractor.hpp"
#include "processing/image/feature/extractor/orb_extractor.hpp"
#include "processing/image/feature/extractor/sift_extractor.hpp"

namespace processing::image {
    std::shared_ptr<FeatureExtractor> FeatureExtractor::create(const std::string_view type) {
        static const auto extractorMap =
                std::unordered_map<std::string_view, std::function<std::shared_ptr<FeatureExtractor>()>>{
                        {"sift", [] { return std::make_shared<SIFTExtractor>(); }},
                        {"akaze", [] { return std::make_shared<AKAZEExtractor>(); }},
                        {"orb", [] { return std::make_shared<ORBExtractor>(); }}};

        auto detector_type =
                type.empty() ? config::get<std::string>("image.feature.extractor.type", "sift") : std::string(type);

        std::ranges::transform(detector_type, detector_type.begin(),
                               [](const unsigned char c) { return std::tolower(c); });

        LOG_DEBUG("Feature detector type: {}", detector_type);

        if (const auto it = extractorMap.find(detector_type); it != extractorMap.end()) {
            LOG_INFO("Using {} feature extractor.", detector_type);
            return it->second();
        }

        LOG_WARN("Invalid feature extractor type '{}', defaulting to SIFT.", detector_type);
        return extractorMap.at("sift")();
    }
} // namespace processing::image
