// File: processing/image/comparator.hpp

#ifndef IMAGE_COMPARATOR_HPP
#define IMAGE_COMPARATOR_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "types/image.hpp"

namespace processing::image {

    class ImageComparator {
    public:
        virtual ~ImageComparator() = default;

        // Compare two images and return a score indicating their similarity.
        // Lower scores indicate more similar images.
        [[nodiscard]] virtual double compare(const cv::Mat &image1, const cv::Mat &image2) const = 0;
        [[nodiscard]] virtual double compare(const Image<> &image1, const Image<> &image2) const = 0;

    protected:
        // Default maximum value to indicate errors.
        static constexpr double error_score_ = std::numeric_limits<double>::max();
    };

} // namespace processing::image

#endif // IMAGE_COMPARATOR_HPP
