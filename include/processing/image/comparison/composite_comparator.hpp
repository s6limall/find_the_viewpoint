// File: processing/image/composite_comparator.hpp

#ifndef COMPOSITE_COMPARATOR_HPP
#define COMPOSITE_COMPARATOR_HPP

#include <algorithm>
#include <cmath>
#include <execution>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "feature_comparator.hpp"
#include "processing/image/comparator.hpp"
#include "processing/image/feature/extractor.hpp"
#include "processing/image/feature/matcher.hpp"
#include "ssim_comparator.hpp"

namespace processing::image {

    class CompositeComparator final : public ImageComparator {
    public:
        struct Config {
            double complexity_scale;
            double base_ssim_weight;
            double edge_low_threshold;
            double edge_high_threshold;
            double feature_weight_factor;
            size_t complexity_sample_size;

            Config() :
                complexity_scale(5.0), base_ssim_weight(0.5), edge_low_threshold(100.0), edge_high_threshold(200.0),
                feature_weight_factor(1.5), complexity_sample_size(1000) {}
        };

        CompositeComparator(std::shared_ptr<FeatureExtractor> extractor, std::shared_ptr<FeatureMatcher> matcher,
                            const Config &config = Config()) :
            config_(config), ssim_comparator_(std::make_shared<SSIMComparator>()),
            feature_comparator_(std::make_shared<FeatureComparator>(std::move(extractor), std::move(matcher))) {}

        CompositeComparator(std::shared_ptr<ImageComparator> ssim_comparator,
                            std::shared_ptr<ImageComparator> feature_comparator, const Config &config = Config()) :
            config_(config), ssim_comparator_(std::move(ssim_comparator)),
            feature_comparator_(std::move(feature_comparator)) {}

        [[nodiscard]] double compare(const cv::Mat &image1, const cv::Mat &image2) const override {
            return computeWeightedScore(image1, image2);
        }

        [[nodiscard]] double compare(const Image<> &image1, const Image<> &image2) const override {
            return computeWeightedScore(image1.getImage(), image2.getImage());
        }

    private:
        Config config_;
        std::shared_ptr<ImageComparator> ssim_comparator_;
        std::shared_ptr<ImageComparator> feature_comparator_;

        [[nodiscard]] double computeWeightedScore(const cv::Mat &image1, const cv::Mat &image2) const {
            auto [ssim_score, feature_score] = std::invoke([&] {
                try {
                    return std::make_pair(ssim_comparator_->compare(image1, image2),
                                          feature_comparator_->compare(image1, image2));
                } catch (const std::exception &e) {
                    LOG_ERROR("Error in component comparators: {}", e.what());
                    return std::make_pair(0.0, 0.0);
                }
            });

            double avg_complexity = (calculateImageComplexity(image1) + calculateImageComplexity(image2)) / 2.0;
            double ssim_weight = config_.base_ssim_weight * (1.0 - avg_complexity);
            double feature_weight = (1.0 - ssim_weight) * config_.feature_weight_factor;

            double weighted_sum = ssim_score * ssim_weight + feature_score * feature_weight;
            double total_weight = ssim_weight + feature_weight;

            // Normalize the score using a sigmoid function
            double normalized_score = 1.0 / (1.0 + std::exp(-weighted_sum + total_weight));

            LOG_INFO("Composite Score: {:.4f} (SSIM: {:.4f} * {:.2f}, Feature: {:.4f} * {:.2f}, Complexity: {:.2f})",
                     normalized_score, ssim_score, ssim_weight / total_weight, feature_score,
                     feature_weight / total_weight, avg_complexity);

            return normalized_score;
        }

        [[nodiscard]] double calculateImageComplexity(const cv::Mat &image) const {
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

            std::vector<cv::Point> samples(config_.complexity_sample_size);
            cv::randu(samples, cv::Scalar(0, 0), cv::Scalar(gray.cols, gray.rows));

            cv::Mat edges;
            cv::Canny(gray, edges, config_.edge_low_threshold, config_.edge_high_threshold);

            size_t edge_count = std::count_if(std::execution::par_unseq, samples.begin(), samples.end(),
                                              [&edges](const cv::Point &pt) { return edges.at<uchar>(pt) > 0; });

            return std::min(static_cast<double>(edge_count) / config_.complexity_sample_size * config_.complexity_scale,
                            1.0);
        }
    };

} // namespace processing::image

#endif // COMPOSITE_COMPARATOR_HPP
