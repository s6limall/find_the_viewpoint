// File: processing/image/comparator.cpp

#include "processing/image/comparator.hpp"
#include "processing/image/comparison/composite_comparator.hpp"
#include "processing/image/comparison/feature_comparator.hpp"
#include "processing/image/comparison/ssim_comparator.hpp"

namespace processing::image {

    using FactoryFunction = std::function<std::shared_ptr<ImageComparator>(const std::shared_ptr<FeatureExtractor> &,
                                                                           const std::shared_ptr<FeatureMatcher> &)>;


    std::pair<std::shared_ptr<ImageComparator>, double>
    ImageComparator::create(const std::shared_ptr<FeatureExtractor> &extractor,
                            const std::shared_ptr<FeatureMatcher> &matcher) {

        static const std::unordered_map<std::string_view, FactoryFunction> map{
                {"ssim", [](auto &, auto &) { return std::make_shared<processing::image::SSIMComparator>(); }},
                {"feature",
                 [](auto &e, auto &m) { return std::make_shared<processing::image::FeatureComparator>(e, m); }},
                {"composite",
                 [](auto &e, auto &m) { return std::make_shared<processing::image::CompositeComparator>(e, m); }}};


        auto comparator = config::get<std::string>("image.comparator.type", "ssim");
        std::ranges::transform(comparator, comparator.begin(), ::tolower);

        const auto &factory =
                map.contains(comparator)
                        ? map.at(comparator)
                        : (LOG_WARN("Invalid comparator type '{}', defaulting to SSIM.", comparator), map.at("ssim"));

        const auto target_score = config::get("image.comparator." + comparator + ".threshold", 0.80);
        LOG_INFO("Using {} image comparator.", comparator);

        return std::make_pair(factory(extractor, matcher), target_score);
    }

} // namespace processing::image
