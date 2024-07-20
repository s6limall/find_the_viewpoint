#ifndef RPCA_HPP
#define RPCA_HPP

#include <cmath>
#include <opencv2/opencv.hpp>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include "common/logging/logger.hpp"
#include "common/utilities/image.hpp"

template<typename T = double>
class RPCA {
    static_assert(std::is_floating_point_v<T>, "Template parameter must be a floating-point type.");

public:
    constexpr explicit RPCA(T tolerance = 1e-7, int max_iterations = 1000) noexcept :
        tolerance_(tolerance), max_iterations_(max_iterations) {}

    [[nodiscard]] auto decompose(const cv::Mat &input_matrix) const -> std::tuple<cv::Mat, cv::Mat> {
        if (input_matrix.empty()) {
            LOG_ERROR("Input matrix is empty.");
            throw std::invalid_argument("Input matrix is empty.");
        }

        cv::Mat input_converted;
        input_matrix.convertTo(input_converted, CV_64F);

        if (input_converted.channels() == 1) {
            return decomposeSingleChannel(input_converted);
        } else {
            std::vector<cv::Mat> channels;
            cv::split(input_converted, channels);

            std::vector<cv::Mat> low_rank_channels, sparse_channels;

            for (const auto &channel: channels) {
                auto [low_rank, sparse] = decomposeSingleChannel(channel);
                low_rank_channels.push_back(low_rank);
                sparse_channels.push_back(sparse);
            }

            cv::Mat low_rank_matrix, sparse_matrix;
            cv::merge(low_rank_channels, low_rank_matrix);
            cv::merge(sparse_channels, sparse_matrix);

            return {low_rank_matrix, sparse_matrix};
        }
    }

private:
    [[nodiscard]] auto decomposeSingleChannel(const cv::Mat &input_matrix) const -> std::tuple<cv::Mat, cv::Mat> {
        const int rows = input_matrix.rows;
        const int cols = input_matrix.cols;
        const T lambda = T(1.0) / std::sqrt(std::max(rows, cols));

        cv::Mat low_rank_matrix = cv::Mat::zeros(rows, cols, CV_64F);
        cv::Mat sparse_matrix = cv::Mat::zeros(rows, cols, CV_64F);
        cv::Mat dual_matrix = cv::Mat::zeros(rows, cols, CV_64F);
        cv::Mat norm_input = input_matrix.clone();
        dual_matrix = input_matrix / std::max(cv::norm(input_matrix, cv::NORM_INF), T(1) / lambda);

        T mu = T(1.25) / cv::norm(input_matrix, cv::NORM_INF);
        const T rho = T(1.5);
        const T norm_input_value = cv::norm(input_matrix, cv::NORM_L2);
        T error = tolerance_ + 1;

        int iteration = 0;
        while (error > tolerance_ && iteration < max_iterations_) {
            low_rank_matrix = singularValueThreshold(input_matrix - sparse_matrix + dual_matrix / mu, T(1) / mu);
            sparse_matrix = shrink(input_matrix - low_rank_matrix + dual_matrix / mu, lambda / mu);
            dual_matrix = dual_matrix + mu * (input_matrix - low_rank_matrix - sparse_matrix);
            mu = rho * mu;
            error = cv::norm(input_matrix - low_rank_matrix - sparse_matrix) / norm_input_value;
            iteration++;
        }

        if (error <= tolerance_) {
            LOG_INFO("Converged in {} iterations.", iteration);
        } else {
            LOG_WARN("Reached maximum iterations ({}) without full convergence. Error: {}", max_iterations_, error);
        }

        return {low_rank_matrix, sparse_matrix};
    }

    [[nodiscard]] static auto singularValueThreshold(const cv::Mat &matrix, T tau) noexcept -> cv::Mat {
        cv::Mat singular_values, left_singular_vectors, right_singular_vectors_transposed;
        cv::SVDecomp(matrix, singular_values, left_singular_vectors, right_singular_vectors_transposed);
        const cv::Mat thresholded_singular_values = cv::Mat::diag(shrink(singular_values, tau));

        return left_singular_vectors * thresholded_singular_values * right_singular_vectors_transposed;
    }

    [[nodiscard]] static auto shrink(const cv::Mat &matrix, T tau) noexcept -> cv::Mat {
        cv::Mat result;
        cv::threshold(cv::abs(matrix) - tau, result, 0, tau, cv::THRESH_TOZERO);
        return sign(matrix).mul(result);
    }

    [[nodiscard]] static auto sign(const cv::Mat &matrix) noexcept -> cv::Mat {
        cv::Mat sign_matrix = cv::Mat::zeros(matrix.size(), matrix.type());
        sign_matrix.setTo(1, matrix > 0);
        sign_matrix.setTo(-1, matrix < 0);
        return sign_matrix;
    }

    const T tolerance_;
    const int max_iterations_;
};

#endif // RPCA_HPP
