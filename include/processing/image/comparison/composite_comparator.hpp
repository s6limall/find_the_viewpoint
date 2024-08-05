// File: processing/image/composite_comparator.hpp

#ifndef COMPOSITE_COMPARATOR_HPP
#define COMPOSITE_COMPARATOR_HPP

#include <algorithm>
#include <cmath>
#include <execution>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "processing/image/comparator.hpp"
#include "processing/image/comparison/color_histogram_compositor.hpp"
#include "processing/image/comparison/feature_comparator.hpp"
#include "processing/image/comparison/ssim_comparator.hpp"
#include "processing/image/feature/extractor.hpp"
#include "processing/image/feature/matcher.hpp"

namespace processing::image {

    class CompositeComparator final : public ImageComparator {
    public:
        struct Config {
            double base_ssim_weight;
            double feature_weight;
            double color_weight;
            double edge_low_threshold;
            double edge_high_threshold;
            size_t complexity_sample_size;

            Config() :
                base_ssim_weight(0.3), feature_weight(0.4), color_weight(0.3), edge_low_threshold(50.0),
                edge_high_threshold(150.0), complexity_sample_size(2000) {}
        };

        CompositeComparator(std::shared_ptr<FeatureExtractor> extractor, std::shared_ptr<FeatureMatcher> matcher,
                            const Config &config = Config()) :
            config_(config), ssim_comparator_(std::make_shared<SSIMComparator>()),
            feature_comparator_(std::make_shared<FeatureComparator>(std::move(extractor), std::move(matcher))),
            color_comparator_(std::make_shared<ColorHistogramComparator>()) {}

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
        std::shared_ptr<ImageComparator> color_comparator_;

        [[nodiscard]] double computeWeightedScore(const cv::Mat &image1, const cv::Mat &image2) const {
            auto [ssim_score, feature_score, color_score] = std::invoke([&] {
                try {
                    return std::make_tuple(ssim_comparator_->compare(image1, image2),
                                           feature_comparator_->compare(image1, image2),
                                           color_comparator_->compare(image1, image2));
                } catch (const std::exception &e) {
                    LOG_ERROR("Error in component comparators: {}", e.what());
                    return std::make_tuple(0.0, 0.0, 0.0);
                }
            });

            double avg_complexity = (calculateImageComplexity(image1) + calculateImageComplexity(image2)) / 2.0;

            double ssim_weight = config_.base_ssim_weight * (1.0 - avg_complexity);
            double feature_weight = config_.feature_weight * (1.0 + avg_complexity);
            double color_weight = config_.color_weight;

            double total_weight = ssim_weight + feature_weight + color_weight;
            ssim_weight /= total_weight;
            feature_weight /= total_weight;
            color_weight /= total_weight;

            double weighted_score =
                    ssim_score * ssim_weight + feature_score * feature_weight + color_score * color_weight;

            LOG_INFO("Composite Score: {:.4f} (SSIM: {:.4f} * {:.2f}, Feature: {:.4f} * {:.2f}, Color: {:.4f} * "
                     "{:.2f}, Complexity: {:.2f})",
                     weighted_score, ssim_score, ssim_weight, feature_score, feature_weight, color_score, color_weight,
                     avg_complexity);

            return weighted_score;
        }

        [[nodiscard]] double calculateImageComplexity(const cv::Mat &image) const {
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

            cv::Mat edges;
            cv::Canny(gray, edges, config_.edge_low_threshold, config_.edge_high_threshold);

            std::vector<cv::Point> samples(config_.complexity_sample_size);
            cv::randu(samples, cv::Scalar(0, 0), cv::Scalar(gray.cols, gray.rows));

            size_t edge_count = std::count_if(std::execution::par_unseq, samples.begin(), samples.end(),
                                              [&edges](const cv::Point &pt) { return edges.at<uchar>(pt) > 0; });

            return static_cast<double>(edge_count) / config_.complexity_sample_size;
        }
    };

} // namespace processing::image

#endif // COMPOSITE_COMPARATOR_HPP
