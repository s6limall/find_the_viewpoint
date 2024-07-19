// File: processing/image/preprocessor.hpp

#ifndef PREPROCESSOR_HPP
#define PREPROCESSOR_HPP

#include <vector>
#include "filtering/image/bilateral_filter.hpp"
#include "filtering/image/rpca.hpp"

namespace processing::image {

    template<typename T = double>
    class Preprocessor {
        static_assert(std::is_floating_point_v<T>, "Template parameter must be a floating-point type.");

    public:
        constexpr explicit Preprocessor(T bilateral_scaling_factor = 30.0, T bilateral_min_diameter = 5.0,
                                        T bilateral_max_diameter = 30.0, T rpca_tolerance = 1e-7,
                                        int rpca_max_iterations = 1000) noexcept;

        [[nodiscard]] auto preprocess(const cv::Mat &input_image, int pyramid_levels = 3) const -> std::vector<cv::Mat>;

    private:
        [[nodiscard]] auto reduceNoise(const cv::Mat &input_image) const -> cv::Mat;

        [[nodiscard]] auto extractForeground(const cv::Mat &input_image) const -> cv::Mat;

        [[nodiscard]] static auto buildImagePyramid(const cv::Mat &image, int levels) noexcept -> std::vector<cv::Mat>;

        BilateralFilter<T> bilateral_filter_;
        RPCA<T> rpca_;
    };

    template<typename T>
    constexpr Preprocessor<T>::Preprocessor(T bilateral_scaling_factor, T bilateral_min_diameter,
                                            T bilateral_max_diameter, T rpca_tolerance,
                                            int rpca_max_iterations) noexcept :
        bilateral_filter_(bilateral_scaling_factor, bilateral_min_diameter, bilateral_max_diameter),
        rpca_(rpca_tolerance, rpca_max_iterations) {}

    template<typename T>
    auto Preprocessor<T>::preprocess(const cv::Mat &input_image, int pyramid_levels) const -> std::vector<cv::Mat> {
        if (input_image.empty()) {
            throw std::invalid_argument("Input image is empty.");
        }

        try {
            const auto noise_reduced_image = reduceNoise(input_image);
            const auto foreground_image = extractForeground(noise_reduced_image);
            return buildImagePyramid(foreground_image, pyramid_levels);
        } catch (const std::exception &e) {
            LOG_ERROR("Preprocessing failed: {}", e.what());
            throw std::runtime_error(std::string("Preprocessing failed: ") + e.what());
        }
    }

    template<typename T>
    auto Preprocessor<T>::reduceNoise(const cv::Mat &input_image) const -> cv::Mat {
        LOG_DEBUG("Applying bilateral filter to reduce noise.");
        return bilateral_filter_.apply(input_image);
    }

    template<typename T>
    auto Preprocessor<T>::extractForeground(const cv::Mat &input_image) const -> cv::Mat {
        LOG_DEBUG("Extracting foreground using Robust PCA.");
        auto [low_rank_matrix, foreground_image] = rpca_.decompose(input_image);
        return foreground_image;
    }

    template<typename T>
    auto Preprocessor<T>::buildImagePyramid(const cv::Mat &image, const int levels) noexcept -> std::vector<cv::Mat> {
        LOG_DEBUG("Building image pyramid with {} levels.", levels);
        return common::utilities::toPyramid(image, levels);
    }


} // namespace processing::image

#endif // PREPROCESSOR_HPP
