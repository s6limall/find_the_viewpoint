// File: processing/image/comparison/ssim_comparator.hpp

#ifndef IMAGE_COMPARATOR_SSIM_HPP
#define IMAGE_COMPARATOR_SSIM_HPP

#include <execution>
#include <opencv2/opencv.hpp>
#include <opencv2/quality/qualityssim.hpp>
#include "processing/image/comparator.hpp"

namespace processing::image {

    class SSIMComparator final : public ImageComparator {
    public:
        SSIMComparator() = default;

        [[nodiscard]] double compare(const cv::Mat &image1, const cv::Mat &image2) const override {
            try {
                if (!isValidInput(image1, image2)) {
                    LOG_WARN("Images are empty, or their sizes/types do not match.");
                    return error_score_;
                }
                return computeSSIM(image1, image2);
            } catch (const std::exception &e) {
                LOG_ERROR("An error occurred during SSIM comparison: {}", e.what());
                return error_score_;
            } catch (...) {
                LOG_ERROR("Unknown error occurred during SSIM comparison.");
                return error_score_;
            }
        }

        [[nodiscard]] double compare(const Image<> &img1, const Image<> &img2) const override {
            return compare(img1.getImage(), img2.getImage());
        }

    private:
        [[nodiscard]] static bool isValidInput(const cv::Mat &image1, const cv::Mat &image2) noexcept {
            if (image1.empty() || image2.empty()) {
                LOG_ERROR("Input images are empty.");
                return false;
            }
            if (image1.size() != image2.size()) {
                LOG_ERROR("Images must have the same size.");
                return false;
            }
            if (image1.type() != image2.type()) {
                LOG_ERROR("Images must have the same type.");
                return false;
            }
            return true;
        }

        [[nodiscard]] static double computeSSIM(const cv::Mat &img1, const cv::Mat &img2) noexcept {
            LOG_TRACE("Computing SSIM.");
            if (img1.channels() == 1) {
                return cv::quality::QualitySSIM::compute(img1, img2, cv::noArray())[0];
            }

            std::vector<cv::Mat> img1_channels, img2_channels;
            cv::split(img1, img1_channels);
            cv::split(img2, img2_channels);

            auto ssim_computer = [&](const cv::Mat &ch1, const cv::Mat &ch2) {
                return cv::quality::QualitySSIM::compute(ch1, ch2, cv::noArray())[0];
            };

            const double ssim_sum =
                    std::transform_reduce(std::execution::par_unseq, img1_channels.begin(), img1_channels.end(),
                                          img2_channels.begin(), 0.0, std::plus<>(), ssim_computer);

            return ssim_sum / img1_channels.size();
        }
    };

} // namespace processing::image

#endif // IMAGE_COMPARATOR_SSIM_HPP
