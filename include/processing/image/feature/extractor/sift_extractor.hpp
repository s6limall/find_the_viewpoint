// File: processing/image/feature/extractor/sift_extractor.hpp

#ifndef SIFT_EXTRACTOR_HPP
#define SIFT_EXTRACTOR_HPP

#include <algorithm>
#include <execution>
#include <opencv2/features2d.hpp>
#include "processing/image/feature/extractor.hpp"

namespace processing::image {

    class SIFTExtractor final : public FeatureExtractor {
    public:
        struct Config {
            int nfeatures;
            int nOctaveLayers;
            double contrastThreshold;
            double edgeThreshold;
            double sigma;
            float response_threshold;
            int max_keypoints;

            Config() :
                nfeatures(0), nOctaveLayers(4), contrastThreshold(0.03), edgeThreshold(15), sigma(1.6),
                response_threshold(0.005f), max_keypoints(3000) {}
        };

        explicit SIFTExtractor(const Config &config = Config());

        [[nodiscard]] std::pair<KeyPoints, Descriptors> extract(const cv::Mat &image) const noexcept override;

    private:
        cv::Ptr<cv::SIFT> sift_;
        Config config_;

        [[nodiscard]] std::pair<KeyPoints, Descriptors> filterKeypoints(const KeyPoints &keypoints,
                                                                        const Descriptors &descriptors) const noexcept;
    };

    inline SIFTExtractor::SIFTExtractor(const Config &config) : config_(config) {
        sift_ = cv::SIFT::create(config_.nfeatures, config_.nOctaveLayers, config_.contrastThreshold,
                                 config_.edgeThreshold, config_.sigma);
    }

    inline std::pair<KeyPoints, Descriptors> SIFTExtractor::extract(const cv::Mat &image) const noexcept {
        try {
            KeyPoints keypoints;
            Descriptors descriptors;

            sift_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

            return filterKeypoints(keypoints, descriptors);
        } catch (const cv::Exception &e) {
            LOG_ERROR("OpenCV exception in SIFT feature extraction: {}", e.what());
            return {{}, {}};
        } catch (const std::exception &e) {
            LOG_ERROR("Exception in SIFT feature extraction: {}", e.what());
            return {{}, {}};
        }
    }

    inline std::pair<KeyPoints, Descriptors>
    SIFTExtractor::filterKeypoints(const KeyPoints &keypoints, const Descriptors &descriptors) const noexcept {
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
            filtered_keypoints.reserve(std::min(config_.max_keypoints, static_cast<int>(keypoints.size())));

            for (size_t i: indices) {
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

#endif // SIFT_EXTRACTOR_HPP
