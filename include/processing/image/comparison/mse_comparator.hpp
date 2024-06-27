// File: processing/image/comparison/mse_comparator.hpp

#ifndef IMAGE_COMPARATOR_MSE_HPP
#define IMAGE_COMPARATOR_MSE_HPP

#include "processing/image/comparator.hpp"

namespace processing::image {
    // Mean Squared Error (MSE) comparator.
    class MSEComparator : public ImageComparator {
    public:
        [[nodiscard]] double compare(const cv::Mat &image1, const cv::Mat &image2) const override;
    };
}

#endif //IMAGE_COMPARATOR_MSE_HPP
