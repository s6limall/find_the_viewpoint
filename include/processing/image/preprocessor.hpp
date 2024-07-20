// File: processing/image/preprocessor.hpp

#ifndef PREPROCESSOR_HPP
#define PREPROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "filtering/image/bilateral_filter.hpp"
#include "filtering/image/rpca.hpp"

namespace processing::image {

    template<typename T = double>
    class Preprocessor {
        static_assert(std::is_floating_point_v<T>, "Template parameter must be a floating-point type.");

    public:
        static auto preprocess(const cv::Mat &input_image, int pyramid_levels = 3,
                               std::optional<T> bilateral_scaling_factor = std::nullopt,
                               std::optional<T> bilateral_min_diameter = std::nullopt,
                               std::optional<T> bilateral_max_diameter = std::nullopt,
                               std::optional<T> rpca_tolerance = std::nullopt,
                               std::optional<int> rpca_max_iterations = std::nullopt)
                -> std::vector<std::vector<cv::Mat>>;

        Preprocessor(const Preprocessor &) = delete;
        Preprocessor &operator=(const Preprocessor &) = delete;
        Preprocessor(Preprocessor &&) = delete;
        Preprocessor &operator=(Preprocessor &&) = delete;

    private:
        Preprocessor(T bilateral_scaling_factor, T bilateral_min_diameter, T bilateral_max_diameter, T rpca_tolerance,
                     int rpca_max_iterations) noexcept;

        [[nodiscard]] auto reduceNoise(const cv::Mat &input_image) const -> cv::Mat;
        [[nodiscard]] auto extractForeground(const cv::Mat &input_image) const -> cv::Mat;
        [[nodiscard]] static auto buildImagePyramid(const cv::Mat &image, int levels) noexcept
                -> std::vector<std::vector<cv::Mat>>;

        BilateralFilter<T> bilateral_filter_;
        RPCA<T> rpca_;

        static std::unique_ptr<Preprocessor> instance_;
        static std::once_flag init_flag_;

        static void initialize(T bilateral_scaling_factor, T bilateral_min_diameter, T bilateral_max_diameter,
                               T rpca_tolerance, int rpca_max_iterations);
    };

    template<typename T>
    std::once_flag Preprocessor<T>::init_flag_;

    template<typename T>
    std::unique_ptr<Preprocessor<T>> Preprocessor<T>::instance_ = nullptr;

    template<typename T>
    Preprocessor<T>::Preprocessor(T bilateral_scaling_factor, T bilateral_min_diameter, T bilateral_max_diameter,
                                  T rpca_tolerance, int rpca_max_iterations) noexcept :
        bilateral_filter_(bilateral_scaling_factor, bilateral_min_diameter, bilateral_max_diameter),
        rpca_(rpca_tolerance, rpca_max_iterations) {}

    template<typename T>
    void Preprocessor<T>::initialize(T bilateral_scaling_factor, T bilateral_min_diameter, T bilateral_max_diameter,
                                     T rpca_tolerance, int rpca_max_iterations) {
        instance_ = std::unique_ptr<Preprocessor>(new Preprocessor(bilateral_scaling_factor, bilateral_min_diameter,
                                                                   bilateral_max_diameter, rpca_tolerance,
                                                                   rpca_max_iterations));
    }

    template<typename T>
    auto Preprocessor<T>::preprocess(const cv::Mat &input_image, int pyramid_levels,
                                     std::optional<T> bilateral_scaling_factor, std::optional<T> bilateral_min_diameter,
                                     std::optional<T> bilateral_max_diameter, std::optional<T> rpca_tolerance,
                                     std::optional<int> rpca_max_iterations) -> std::vector<std::vector<cv::Mat>> {

        std::call_once(init_flag_, [&] {
            initialize(bilateral_scaling_factor.value_or(30.0), bilateral_min_diameter.value_or(5.0),
                       bilateral_max_diameter.value_or(30.0), rpca_tolerance.value_or(1e-7),
                       rpca_max_iterations.value_or(1000));
        });

        if (input_image.empty()) {
            throw std::invalid_argument("Input image is empty.");
        }

        LOG_INFO("Input image size: ({}, {}), type: {}", input_image.rows, input_image.cols, input_image.type());

        // Convert input image to floating-point format based on its depth
        cv::Mat float_image;
        double normalization_factor;
        switch (input_image.depth()) {
            case CV_8U:
                normalization_factor = 1.0 / 255.0;
                input_image.convertTo(float_image, CV_32F, normalization_factor);
                break;
            case CV_16U:
                normalization_factor = 1.0 / 65535.0;
                input_image.convertTo(float_image, CV_32F, normalization_factor);
                break;
            case CV_32F:
                normalization_factor = 1.0;
                input_image.convertTo(float_image, CV_32F, normalization_factor);
                break;
            case CV_64F:
                normalization_factor = 1.0;
                input_image.convertTo(float_image, CV_64F, normalization_factor);
                break;
            default:
                throw std::invalid_argument("Unsupported image depth.");
        }

        try {
            const auto noise_reduced_image = instance_->reduceNoise(float_image);
            return buildImagePyramid(noise_reduced_image, pyramid_levels);
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
        auto input_image_gray = common::utilities::toGrayscale(input_image);
        if (input_image_gray.type() != CV_32F && input_image_gray.type() != CV_64F) {
            LOG_ERROR("Input image type is not float or double. Current type: {}", input_image_gray.type());
            throw std::runtime_error("Invalid input image type for RPCA.");
        }

        auto [low_rank_matrix, foreground_image] = rpca_.decompose(input_image_gray);
        return foreground_image;
    }

    template<typename T>
    auto Preprocessor<T>::buildImagePyramid(const cv::Mat &image, const int levels) noexcept
            -> std::vector<std::vector<cv::Mat>> {
        LOG_DEBUG("Building image pyramid with {} levels.", levels);
        std::vector<std::vector<cv::Mat>> pyramids(2);
        std::vector<cv::Mat> &gaussianPyramid = pyramids[0];
        std::vector<cv::Mat> &laplacianPyramid = pyramids[1];

        // Reserve space for the pyramids
        gaussianPyramid.reserve(levels);
        laplacianPyramid.reserve(levels - 1);

        // Add the original image to the Gaussian pyramid
        gaussianPyramid.push_back(image);

        cv::Mat currentImage = image;
        for (int i = 1; i < levels; ++i) {
            cv::Mat down, up, laplacian;
            cv::pyrDown(currentImage, down);
            gaussianPyramid.push_back(down);

            // Upsample the downsampled image
            cv::pyrUp(down, up, currentImage.size());

            // Ensure the sizes match before subtracting
            if (up.size() != currentImage.size()) {
                cv::resize(up, up, currentImage.size());
            }

            // Calculate the Laplacian
            laplacian = currentImage - up;
            laplacianPyramid.push_back(laplacian);

            currentImage = down;
        }

        return pyramids;
    }

} // namespace processing::image

#endif // PREPROCESSOR_HPP
