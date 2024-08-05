// File: processing/image/feature/extractor/orb_extractor.hpp

#ifndef ORB_EXTRACTOR_HPP
#define ORB_EXTRACTOR_HPP

#include <algorithm>
#include <execution>
#include <opencv2/features2d.hpp>
#include "processing/image/feature/extractor.hpp"

namespace processing::image {

    class ORBExtractor final : public FeatureExtractor {
    public:
        struct Config {
            int nfeatures;
            float scaleFactor;
            int nlevels;
            int edgeThreshold;
            int firstLevel;
            int WTA_K;
            cv::ORB::ScoreType scoreType;
            int patchSize;
            int fastThreshold;
            float response_threshold;
            int max_keypoints;

            Config() :
                nfeatures(3000), scaleFactor(1.2f), nlevels(8), edgeThreshold(31), firstLevel(0), WTA_K(2),
                scoreType(cv::ORB::FAST_SCORE), patchSize(31), fastThreshold(20), response_threshold(0.01f),
                max_keypoints(3000) {}
        };

        explicit ORBExtractor(const Config &config = Config());

        [[nodiscard]] std::pair<KeyPoints, Descriptors> extract(const cv::Mat &image) const noexcept override;

    private:
        cv::Ptr<cv::ORB> orb_;
        Config config_;

        [[nodiscard]] std::pair<KeyPoints, Descriptors> filterKeypoints(const KeyPoints &keypoints,
                                                                        const Descriptors &descriptors) const noexcept;
    };

    inline ORBExtractor::ORBExtractor(const Config &config) : config_(config) {
        orb_ = cv::ORB::create(config_.nfeatures, config_.scaleFactor, config_.nlevels, config_.edgeThreshold,
                               config_.firstLevel, config_.WTA_K, config_.scoreType, config_.patchSize,
                               config_.fastThreshold);
    }

    inline std::pair<KeyPoints, Descriptors> ORBExtractor::extract(const cv::Mat &image) const noexcept {
        try {
            KeyPoints keypoints;
            Descriptors descriptors;

            orb_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

            return filterKeypoints(keypoints, descriptors);
        } catch (const cv::Exception &e) {
            LOG_ERROR("OpenCV exception in ORB feature extraction: {}", e.what());
            return {{}, {}};
        } catch (const std::exception &e) {
            LOG_ERROR("Exception in ORB feature extraction: {}", e.what());
            return {{}, {}};
        }
    }

    inline std::pair<KeyPoints, Descriptors>
    ORBExtractor::filterKeypoints(const KeyPoints &keypoints, const Descriptors &descriptors) const noexcept {
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

#endif // ORB_EXTRACTOR_HPP
