// File: processing/vision/sphere_detector.cpp

#include "processing/vision/sphere_detector.hpp"
#include <spdlog/spdlog.h>

namespace processing::vision {
    std::pair<cv::Point2f, float> SphereDetector::detect(const cv::Mat& image) const {
        cv::Mat gray, blurred, edged;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
        cv::Canny(blurred, edged, 50, 150);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(edged, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (contours.empty()) {
            spdlog::warn("No contours detected.");
            return {cv::Point2f(0, 0), 0};
        }

        auto largest_contour = *std::max_element(contours.begin(), contours.end(),
                                                 [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                                                     return cv::contourArea(a) < cv::contourArea(b);
                                                 });

        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(largest_contour, center, radius);

        spdlog::info("Bounding sphere detected: center=({}, {}), radius={}", center.x, center.y, radius);
        return {center, radius};
    }

}
