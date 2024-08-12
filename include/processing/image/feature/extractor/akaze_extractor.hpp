// File: processing/image/feature/extractor/akaze_extractor.hpp

#ifndef AKAZE_EXTRACTOR_HPP
#define AKAZE_EXTRACTOR_HPP

#include <algorithm>
#include <execution>
#include <opencv2/features2d.hpp>
#include "processing/image/feature/extractor.hpp"

namespace processing::image {

    class AKAZEExtractor final : public FeatureExtractor {
    public:
        struct Config {
            cv::AKAZE::DescriptorType descriptor_type;
            int descriptor_size;
            int descriptor_channels;
            float threshold;
            int octaves;
            int octave_layers;
            cv::KAZE::DiffusivityType diffusivity;
            float response_threshold;
            size_t max_keypoints;

            Config() :
                descriptor_type(cv::AKAZE::DESCRIPTOR_MLDB), descriptor_size(0), descriptor_channels(3),
                threshold(0.0003f), octaves(5), octave_layers(5), diffusivity(cv::KAZE::DIFF_PM_G2),
                response_threshold(0.003f), max_keypoints(3000) {}
        };

        explicit AKAZEExtractor(const Config &config = Config());

        [[nodiscard]] std::pair<KeyPoints, Descriptors> extract(const cv::Mat &image) const noexcept override;

    private:
        cv::Ptr<cv::AKAZE> akaze_;
        Config config_;

        [[nodiscard]] std::pair<KeyPoints, Descriptors> filterKeypoints(const KeyPoints &keypoints,
                                                                        const Descriptors &descriptors) const noexcept;
    };

    inline AKAZEExtractor::AKAZEExtractor(const Config &config) : config_(config) {
        akaze_ = cv::AKAZE::create(config_.descriptor_type, config_.descriptor_size, config_.descriptor_channels,
                                   config_.threshold, config_.octaves, config_.octave_layers, config_.diffusivity);
    }

    inline std::pair<KeyPoints, Descriptors> AKAZEExtractor::extract(const cv::Mat &image) const noexcept {
        try {
            KeyPoints keypoints;
            Descriptors descriptors;

            akaze_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

            return filterKeypoints(keypoints, descriptors);
        } catch (const cv::Exception &e) {
            LOG_ERROR("OpenCV exception in AKAZE feature extraction: {}", e.what());
            return {{}, {}};
        } catch (const std::exception &e) {
            LOG_ERROR("Exception in AKAZE feature extraction: {}", e.what());
            return {{}, {}};
        }
    }

    inline std::pair<KeyPoints, Descriptors>
    AKAZEExtractor::filterKeypoints(const KeyPoints &keypoints, const Descriptors &descriptors) const noexcept {
        if (keypoints.empty()) {
            return {keypoints, descriptors};
        }

        try {
            const float max_response =
                    std::ranges::max_element(keypoints, [](const cv::KeyPoint &a, const cv::KeyPoint &b) {
                        return a.response < b.response;
                    })->response;

            const float response_threshold = config_.response_threshold * max_response;

            std::vector<size_t> indices(keypoints.size());
            std::iota(indices.begin(), indices.end(), 0);

            std::ranges::sort(indices, [&keypoints](size_t a, size_t b) {
                return keypoints[a].response > keypoints[b].response;
            });

            KeyPoints filtered_keypoints;
            Descriptors filtered_descriptors;
            filtered_keypoints.reserve(std::min(config_.max_keypoints, keypoints.size()));

            for (const size_t i: indices) {
                if (keypoints[i].response >= response_threshold && filtered_keypoints.size() < config_.max_keypoints) {
                    filtered_keypoints.push_back(keypoints[i]);
                    filtered_descriptors.push_back(descriptors.row(i));
                } else {
                    break;
                }
            }

            return {filtered_keypoints, filtered_descriptors};
        } catch (const std::exception &e) {
            LOG_ERROR("Exception in keypoint filtering: {}", e.what());
            return {keypoints, descriptors};
        }
    }

} // namespace processing::image

#endif // AKAZE_EXTRACTOR_HPP
