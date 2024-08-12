// File: common/utilities/visualizer.hpp

#ifndef COMMON_UTILITIES_VISUALIZER_HPP
#define COMMON_UTILITIES_VISUALIZER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include "common/logging/logger.hpp"
#include "types/image.hpp"

namespace common::utilities {

    class Visualizer {
    public:
        static void diff(const Image<> &target, const Image<> &best_image,
                         const std::string &output_path = "diff.png") {
            cv::Mat layout(target.getImage().rows * 2, target.getImage().cols * 2, CV_8UC3);

            // 1. Target image (top left)
            target.getImage().copyTo(layout(cv::Rect(0, 0, target.getImage().cols, target.getImage().rows)));

            // 2. Best viewpoint image (top right)
            best_image.getImage().copyTo(layout(
                    cv::Rect(target.getImage().cols, 0, best_image.getImage().cols, best_image.getImage().rows)));

            // 3. Overlay of target and best image (bottom left)
            cv::Mat overlay;
            cv::addWeighted(target.getImage(), 0.5, best_image.getImage(), 0.5, 0, overlay);
            overlay.copyTo(layout(cv::Rect(0, target.getImage().rows, target.getImage().cols, target.getImage().rows)));

            // 4. Difference image (bottom right)
            cv::Mat diff;
            cv::absdiff(target.getImage(), best_image.getImage(), diff);
            diff.copyTo(layout(cv::Rect(target.getImage().cols, target.getImage().rows, target.getImage().cols,
                                        target.getImage().rows)));

            // Add labels
            addLabel(layout, "Target", cv::Point(10, 20));
            addLabel(layout, "Best Viewpoint", cv::Point(target.getImage().cols + 10, 20));
            addLabel(layout, "Overlay", cv::Point(10, target.getImage().rows + 20));
            addLabel(layout, "Difference", cv::Point(target.getImage().cols + 10, target.getImage().rows + 20));

            cv::imwrite(output_path, layout);
            LOG_INFO("Visualization saved to: {}", output_path);
        }

    private:
        static void addLabel(cv::Mat &image, const std::string &label, const cv::Point &position) {
            cv::putText(image, label, position, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            cv::putText(image, label, position, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
    };

} // namespace common::utilities

#endif // COMMON_UTILITIES_VISUALIZER_HPP
