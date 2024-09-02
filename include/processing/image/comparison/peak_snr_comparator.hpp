// File: processing/image/comparison/peak_snr_comparator.hpp

#ifndef PEAK_SNR_COMPARATOR_HPP
#define PEAK_SNR_COMPARATOR_HPP

#include <cmath>
#include <execution>
#include <limits>
#include <opencv2/core.hpp>
#include <stdexcept>
#include "common/logging/logger.hpp"
#include "types/image.hpp"

namespace processing::image {

    class PeakSNRComparator {
    public:
        PeakSNRComparator() = default;

        [[nodiscard]] static double compare(const cv::Mat &image1, const cv::Mat &image2) {
            if (!isValidInput(image1, image2)) {
                LOG_WARN("Invalid input images for PSNR comparison.");
                return 0.0;
            }
            return computePSNR(image1, image2);
        }

        [[nodiscard]] static double compare(const Image<> &img1, const Image<> &img2) {
            return compare(img1.getImage(), img2.getImage());
        }

    private:
        [[nodiscard]] static bool isValidInput(const cv::Mat &image1, const cv::Mat &image2) noexcept {
            return !image1.empty() && !image2.empty() && image1.size() == image2.size() &&
                   image1.type() == image2.type();
        }

        [[nodiscard]] static double computePSNR(const cv::Mat &img1, const cv::Mat &img2) noexcept {
            const double max_intensity_squared = getMaxIntensitySquared(img1);

            cv::Mat diff;
            cv::absdiff(img1, img2, diff);

            if (img1.depth() != CV_32F && img1.depth() != CV_64F) {
                diff.convertTo(diff, CV_32F);
            }

            // Compute MSE directly with parallel reduction
            const double mse = std::transform_reduce(std::execution::par_unseq, diff.begin<float>(), diff.end<float>(),
                                                     0.0, std::plus<>(), [](const float val) { return val * val; }) /
                               diff.total();

            if (mse <= std::numeric_limits<double>::epsilon()) {
                return 1.0; // Perfect match
            }

            const double psnr = 10.0 * std::log10(max_intensity_squared / mse);
            return normalizePSNR(psnr, img1.depth());
        }

        [[nodiscard]] static double getMaxIntensitySquared(const cv::Mat &image) {
            double max_intensity = 0.0;
            switch (image.depth()) {
                case CV_8U:
                    max_intensity = 255.0;
                    break;
                case CV_16U:
                    max_intensity = 65535.0;
                    break;
                case CV_32F:
                case CV_64F:
                    cv::minMaxLoc(image, nullptr, &max_intensity);
                    break;
                default:
                    throw std::runtime_error("Unsupported image depth for PSNR calculation");
            }
            return max_intensity * max_intensity;
        }

        [[nodiscard]] static double normalizePSNR(const double psnr, const int depth) noexcept {
            constexpr double standard_max_psnr = 100.0;
            double dynamic_max_psnr = standard_max_psnr;

            if (depth == CV_16U) {
                dynamic_max_psnr = standard_max_psnr + 20.0; // change for higher bit depth
            }

            return std::clamp(psnr / dynamic_max_psnr, 0.0, 1.0);
        }
    };

} // namespace processing::image

#endif // PEAK_SNR_COMPARATOR_HPP
