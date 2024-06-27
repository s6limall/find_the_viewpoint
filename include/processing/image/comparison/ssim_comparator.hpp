// File: processing/image/comparison/ssim_comparator.hpp

#ifndef IMAGE_COMPARATOR_SSIM_HPP
#define IMAGE_COMPARATOR_SSIM_HPP

#include "processing/image/comparator.hpp"

namespace processing::image {
    class SSIMComparator : public ImageComparator {
    public:
        [[nodiscard]] double compare(const cv::Mat &image1, const cv::Mat &image2) const override;
    };
}

#endif //IMAGE_COMPARATOR_SSIM_HPP
