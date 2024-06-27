// File: processing/image/comparison/mse_comparator.cpp

#include "processing/image/comparison/mse_comparator.hpp"
#include "config/configuration.hpp"
#include <opencv2/opencv.hpp>
#include <stdexcept>

namespace processing::image {

    double MSEComparator::compare(const cv::Mat &image1, const cv::Mat &image2) const {
        if (image1.empty() || image2.empty()) {
            throw std::invalid_argument("One or both images are empty");
        }
        if (image1.size() != image2.size() || image1.type() != image2.type()) {
            throw std::invalid_argument("Images must have the same size and type");
        }

        const auto& config = config::Configuration::getInstance();
        bool normalize = config.get<bool>("image_comparator.mse.normalize", true);

        cv::Mat diff;
        cv::absdiff(image1, image2, diff);
        diff.convertTo(diff, CV_32F);
        diff = diff.mul(diff);

        double mse = cv::sum(diff)[0];
        if (normalize) {
            mse /= static_cast<double>(image1.total());
        }

        return mse;
    }

}
