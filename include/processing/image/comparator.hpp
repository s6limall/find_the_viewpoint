// File: processing/image/comparator.hpp

#ifndef IMAGE_COMPARATOR_HPP
#define IMAGE_COMPARATOR_HPP

#include <opencv2/opencv.hpp>

namespace processing::image {

    class ImageComparator {
    public:
        virtual ~ImageComparator() = default;

        // Compare two images and return a score indicating their similarity.
        // Lower scores indicate more similar images.
        [[nodiscard]] virtual double compare(const cv::Mat& image1, const cv::Mat& image2) const = 0;
    };
}

#endif //IMAGE_COMPARATOR_HPP
