// File: processing/image/comparison/peak_snr_comparator.hpp

#ifndef PEAK_SNR_COMPARATOR_HPP
#define PEAK_SNR_COMPARATOR_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "processing/image/comparator.hpp"

namespace processing::image {

    class PeakSNRComparator final : public ImageComparator {
    public:
        PeakSNRComparator() = default;

        [[nodiscard]] double compare(const cv::Mat &image1, const cv::Mat &image2) const override {
            if (!isValidInput(image1, image2)) {
                LOG_WARN("Images are empty, or their sizes/types do not match.");
                return error_score_;
            }
            return computePSNR(image1, image2);
        }

        [[nodiscard]] double compare(const Image<> &img1, const Image<> &img2) const override {
            return compare(img1.getImage(), img2.getImage());
        }

    private:
        [[nodiscard]] static inline bool isValidInput(const cv::Mat &image1, const cv::Mat &image2) noexcept {
            return !image1.empty() && !image2.empty() && (image1.size() == image2.size()) &&
                   (image1.type() == image2.type());
        }

        [[nodiscard]] static double computePSNR(const cv::Mat &img1, const cv::Mat &img2) noexcept {
            cv::Mat diff;
            cv::absdiff(img1, img2, diff); // absolute difference
            diff.convertTo(diff, CV_32F);

            // MSE: sum squares, then compute mean
            const double mse = cv::sum(diff.mul(diff))[0] / (static_cast<double>(diff.total()) * diff.channels());

            if (mse <= epsilon_) {
                return error_score_; // Identical images, return error score
            }

            // PSNR calculation
            constexpr double max_i = 255.0;
            return 10.0 * std::log10((max_i * max_i) / mse);
        }
    };

} // namespace processing::image

#endif // PEAK_SNR_COMPARATOR_HPP
