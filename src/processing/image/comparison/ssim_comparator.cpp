// File: processing/image/ssim_calculator.cpp

#include "processing/image/comparison/ssim_comparator.hpp"
#include <opencv2/quality/qualityssim.hpp>
#include <iostream>
#include <numeric>
#include <vector>

namespace processing::image {

    double SSIMComparator::compare(const cv::Mat &image1, const cv::Mat &image2) const {
        try {
            if (!validateImages(image1, image2)) {
                std::cerr << "Invalid input images." << std::endl;
                return error_score_;
            }

            return (image1.channels() == 1) ? computeSSIM(image1, image2) : computeMultiChannelSSIM(image1, image2);
        } catch (const std::exception &e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return error_score_;
        } catch (...) {
            std::cerr << "Unknown error occurred." << std::endl;
            return error_score_;
        }
    }

    double SSIMComparator::computeSSIM(const cv::Mat &img1, const cv::Mat &img2) noexcept {
        const cv::Scalar ssim_value = cv::quality::QualitySSIM::compute(img1, img2, cv::noArray());
        return ssim_value[0];
    }

    double SSIMComparator::computeMultiChannelSSIM(const cv::Mat &img1, const cv::Mat &img2) const noexcept {
        std::vector<cv::Mat> channels1, channels2;
        cv::split(img1, channels1);
        cv::split(img2, channels2);

        const double ssim_total = std::transform_reduce(
                channels1.begin(), channels1.end(), channels2.begin(), 0.0,
                std::plus<>(),
                [this](const cv::Mat &ch1, const cv::Mat &ch2) { return computeSSIM(ch1, ch2); }
                );

        return ssim_total / static_cast<double>(channels1.size());
    }

    bool SSIMComparator::validateImages(const cv::Mat &img1, const cv::Mat &img2) noexcept {
        if (img1.empty() || img2.empty()) {
            std::cerr << "One or both images are empty." << std::endl;
            return false;
        }

        if (img1.size() != img2.size()) {
            std::cerr << "Images must have the same size." << std::endl;
            return false;
        }

        if (img1.type() != img2.type()) {
            std::cerr << "Images must have the same type." << std::endl;
            return false;
        }

        return true;
    }

}
