// File: processing/image/comparison/ssim_comparator.hpp

#ifndef IMAGE_COMPARATOR_SSIM_HPP
#define IMAGE_COMPARATOR_SSIM_HPP

#include <tuple>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "processing/image/comparator.hpp"

namespace processing::image {

    class SSIMComparator final : public ImageComparator {
    public:
        ~SSIMComparator() override = default;

        [[nodiscard]] double compare(const cv::Mat &image1, const cv::Mat &image2) const override;

        [[nodiscard]] double compare(const Image<> &image1, const Image<> &image2) const override;

    private:
        [[nodiscard]] static double computeSSIM(const cv::Mat &img1, const cv::Mat &img2) noexcept;

        [[nodiscard]] double computeMultiChannelSSIM(const cv::Mat &img1, const cv::Mat &img2) const noexcept;

        [[nodiscard]] static bool validateImages(const cv::Mat &img1, const cv::Mat &img2) noexcept;
    };


} // namespace processing::image

#endif // IMAGE_COMPARATOR_SSIM_HPP
