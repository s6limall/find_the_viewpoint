// File: processing/image/color_histogram_comparator.hpp

#ifndef COLOR_HISTOGRAM_COMPARATOR_HPP
#define COLOR_HISTOGRAM_COMPARATOR_HPP

#include <array>
#include <execution>
#include <opencv2/opencv.hpp>
#include "processing/image/comparator.hpp"

namespace processing::image {

    class ColorHistogramComparator final : public ImageComparator {
    public:
        struct Config {
            int color_bins;
            float range_min;
            float range_max;
            int sample_step;

            Config() : color_bins(64), range_min(0.0f), range_max(256.0f), sample_step(4) {}
        };

        explicit ColorHistogramComparator(const Config &config = Config()) : config_(config) {}

        [[nodiscard]] double compare(const cv::Mat &image1, const cv::Mat &image2) const override {
            const auto hist1 = calculateHistogram(image1);
            const auto hist2 = calculateHistogram(image2);

            return compareHistograms(hist1, hist2);
        }

        [[nodiscard]] double compare(const Image<> &image1, const Image<> &image2) const override {
            return compare(image1.getImage(), image2.getImage());
        }

    private:
        Config config_;

        using HistogramType = std::array<cv::Mat, 3>;

        [[nodiscard]] HistogramType calculateHistogram(const cv::Mat &image) const {
            HistogramType hists;
            constexpr std::array<int, 3> channels = {0, 1, 2};
            const int hist_size = config_.color_bins;
            float range[] = {config_.range_min, config_.range_max};
            const float *hist_range = {range};

            cv::Mat mask;
            if (config_.sample_step > 1) {
                mask = cv::Mat::zeros(image.size(), CV_8UC1);
                for (int y = 0; y < image.rows; y += config_.sample_step) {
                    for (int x = 0; x < image.cols; x += config_.sample_step) {
                        mask.at<uchar>(y, x) = 255;
                    }
                }
            }

            for (int i = 0; i < 3; ++i) {
                cv::calcHist(&image, 1, &channels[i], mask, hists[i], 1, &hist_size, &hist_range, true, false);
                cv::normalize(hists[i], hists[i], 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
            }

            return hists;
        }

        [[nodiscard]] static double compareHistograms(const HistogramType &hist1, const HistogramType &hist2) {
            std::array<double, 3> similarities{};
            std::transform(std::execution::par_unseq, hist1.begin(), hist1.end(), hist2.begin(), similarities.begin(),
                           [](const cv::Mat &h1, const cv::Mat &h2) {
                               return cv::compareHist(h1, h2, cv::HISTCMP_BHATTACHARYYA);
                           });

            // heuristic - assigning equal weight to each channel
            constexpr std::array<double, 3> weights = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
            const double distance = std::inner_product(similarities.begin(), similarities.end(), weights.begin(), 0.0);

            // Convert distance to similarity score
            return 1.0 - distance;
        }
    };

} // namespace processing::image

#endif // COLOR_HISTOGRAM_COMPARATOR_HPP
