// File: processing/image/color_histogram_comparator.hpp

#ifndef COLOR_HISTOGRAM_COMPARATOR_HPP
#define COLOR_HISTOGRAM_COMPARATOR_HPP

#include <opencv2/opencv.hpp>
#include "processing/image/comparator.hpp"

namespace processing::image {

    class ColorHistogramComparator final : public ImageComparator {
    public:
        struct Config {
            int histogram_size;
            float range_min;
            float range_max;

            Config() : histogram_size(32), range_min(0.0f), range_max(256.0f) {}
        };

        explicit ColorHistogramComparator(const Config& config = Config()) : config_(config) {}

        [[nodiscard]] double compare(const cv::Mat& image1, const cv::Mat& image2) const override {
            cv::Mat hist1 = calculateHistogram(image1);
            cv::Mat hist2 = calculateHistogram(image2);

            return cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
        }

        [[nodiscard]] double compare(const Image<>& image1, const Image<>& image2) const override {
            return compare(image1.getImage(), image2.getImage());
        }

    private:
        Config config_;

        [[nodiscard]] cv::Mat calculateHistogram(const cv::Mat& image) const {
            cv::Mat hsv;
            cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

            std::vector<cv::Mat> hsv_planes;
            cv::split(hsv, hsv_planes);

            float range[] = { config_.range_min, config_.range_max };
            const float* hist_range = { range };
            bool uniform = true, accumulate = false;

            cv::Mat hist;
            cv::calcHist(&hsv_planes[0], 1, 0, cv::Mat(), hist, 1, &config_.histogram_size, &hist_range, uniform, accumulate);

            cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

            return hist;
        }
    };

} // namespace processing::image

#endif // COLOR_HISTOGRAM_COMPARATOR_HPP
