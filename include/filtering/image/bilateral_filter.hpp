// File: filtering/image/bilateral_filter.hpp

#ifndef BILATERAL_FILTER_HPP
#define BILATERAL_FILTER_HPP

#include <optional>
#include <stdexcept>
#include <type_traits>

#include <opencv2/opencv.hpp>

#include "common/utilities/image.hpp"

template<typename T = double>
class BilateralFilter {
    static_assert(std::is_floating_point_v<T>, "Template parameter must be a floating-point type.");

public:
    constexpr explicit BilateralFilter(T scaling_factor = 30.0, T min_diameter = 5.0, T max_diameter = 30.0) noexcept :
        scaling_factor_(scaling_factor), min_diameter_(min_diameter), max_diameter_(max_diameter) {}

    static auto apply(const cv::Mat &src, std::optional<int> diameter = std::nullopt,
                      std::optional<T> sigma_color = std::nullopt, std::optional<T> sigma_space = std::nullopt,
                      T scaling_factor = 30.0, T min_diameter = 5.0, T max_diameter = 30.0) -> cv::Mat {
        if (src.empty()) {
            throw std::invalid_argument("Input image is empty.");
        }

        if (!diameter || !sigma_color || !sigma_space) {
            std::tie(diameter, sigma_color, sigma_space) =
                    estimateParameters(src, scaling_factor, min_diameter, max_diameter);
        }

        cv::Mat dst;
        cv::bilateralFilter(src, dst, diameter.value(), sigma_color.value(), sigma_space.value());
        return dst;
    }

private:
    static auto estimateParameters(const cv::Mat &src, T scaling_factor, T min_diameter, T max_diameter) noexcept
            -> std::tuple<std::optional<int>, std::optional<T>, std::optional<T>> {
        auto stddev = common::utilities::computeStandardDeviation(src);
        auto diameter = static_cast<int>(std::clamp(scaling_factor * stddev / 100.0, static_cast<T>(min_diameter),
                                                    static_cast<T>(max_diameter)));
        return {diameter, stddev, stddev};
    }

    T scaling_factor_;
    T min_diameter_;
    T max_diameter_;
};


#endif // BILATERAL_FILTER_HPP
