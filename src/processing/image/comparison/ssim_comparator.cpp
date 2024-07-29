// File: processing/image/ssim_calculator.cpp

#include "processing/image/comparison/ssim_comparator.hpp"
#include <iostream>
#include <numeric>
#include <opencv2/quality/qualityssim.hpp>
#include <vector>

#include "common/logging/logger.hpp"

namespace processing::image {

    double SSIMComparator::compare(const Image<> &image1, const Image<> &image2) const {
        return compare(image1.getImage(), image2.getImage());
    }

    double SSIMComparator::compare(const cv::Mat &image1, const cv::Mat &image2) const {
        try {
            if (!validateImages(image1, image2)) {
                LOG_ERROR("Invalid input images.");
                return error_score_;
            }

            return (image1.channels() == 1) ? computeSSIM(image1, image2) : computeMultiChannelSSIM(image1, image2);
        } catch (const std::exception &e) {
            LOG_ERROR("An error occurred during SSIM comparison: {}", e.what());
            return error_score_;
        } catch (...) {
            LOG_ERROR("Unknown error occurred during SSIM comparison.");
            return error_score_;
        }
    }

    double SSIMComparator::computeSSIM(const cv::Mat &img1, const cv::Mat &img2) noexcept {
        LOG_TRACE("Computing SSIM for single channel images.");
        const cv::Scalar ssim_value = cv::quality::QualitySSIM::compute(img1, img2, cv::noArray());
        return ssim_value[0];
    }

    double SSIMComparator::computeMultiChannelSSIM(const cv::Mat &img1, const cv::Mat &img2) const noexcept {
        LOG_TRACE("Computing SSIM for multi-channel images.");
        std::vector<cv::Mat> channels1, channels2;
        cv::split(img1, channels1);
        cv::split(img2, channels2);

        const double ssim_total =
                std::transform_reduce(channels1.begin(), channels1.end(), channels2.begin(), 0.0, std::plus<>(),
                                      [this](const cv::Mat &ch1, const cv::Mat &ch2) { return computeSSIM(ch1, ch2); });

        return ssim_total / static_cast<double>(channels1.size());
    }

    bool SSIMComparator::validateImages(const cv::Mat &img1, const cv::Mat &img2) noexcept {
        if (img1.empty() || img2.empty()) {
            LOG_ERROR("Input images are empty.");
            return false;
        }

        if (img1.size() != img2.size()) {
            LOG_ERROR("Images must have the same size.");
            return false;
        }

        if (img1.type() != img2.type()) {
            LOG_ERROR("Images must have the same type.");
            return false;
        }

        return true;
    }

} // namespace processing::image
