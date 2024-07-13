// File: processing/image/comparison/ssim_comparator.hpp

#ifndef IMAGE_COMPARATOR_SSIM_HPP
#define IMAGE_COMPARATOR_SSIM_HPP

#include <tuple>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "processing/image/comparator.hpp"

namespace processing::image {

    class SSIMComparator final : public ImageComparator {
    public:
        ~SSIMComparator() override = default;

        [[nodiscard]] double compare(const cv::Mat &image1, const cv::Mat &image2) const override;

    private:
        [[nodiscard]] static double computeSSIM(const cv::Mat &img1, const cv::Mat &img2) noexcept;

        [[nodiscard]] double computeMultiChannelSSIM(const cv::Mat &img1, const cv::Mat &img2) const noexcept;

        [[nodiscard]] static bool validateImages(const cv::Mat &img1, const cv::Mat &img2) noexcept;
    };


}

#endif //IMAGE_COMPARATOR_SSIM_HPP
