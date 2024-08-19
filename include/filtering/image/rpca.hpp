#ifndef RPCA_HPP
#define RPCA_HPP

#include <cmath>
#include <concepts>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <tuple>

#include "common/logging/logger.hpp"
#include "types/concepts.hpp"

template<FloatingPoint T = double>
class RPCA {
public:
    constexpr explicit RPCA(T tolerance = 1e-7, const int max_iterations = 1000) noexcept :
        tolerance_(tolerance), max_iterations_(max_iterations) {}

    [[nodiscard]] auto decompose(const cv::Mat &input_matrix) const -> std::tuple<cv::Mat, cv::Mat> {
        if (input_matrix.empty())
            throw std::invalid_argument("Input matrix is empty.");
        cv::Mat converted;
        input_matrix.convertTo(converted, CV_64F);
        return converted.channels() == 1 ? decomposeSingleChannel(converted) : decomposeMultiChannel(converted);
    }

private:
    [[nodiscard]] auto decomposeSingleChannel(const cv::Mat &matrix) const -> std::tuple<cv::Mat, cv::Mat> {
        auto [rows, cols] = std::make_pair(matrix.rows, matrix.cols);
        T lambda = T(1.0) / std::sqrt(std::max(rows, cols));
        cv::Mat low_rank, sparse, dual = matrix / std::max(cv::norm(matrix, cv::NORM_INF), lambda);
        T mu = T(1.25) / cv::norm(matrix, cv::NORM_INF), rho = T(1.5), error = tolerance_ + 1;
        int iteration = 0;

        while (error > tolerance_ && iteration++ < max_iterations_) {
            std::tie(low_rank, sparse) = iterate(matrix, low_rank, sparse, dual, mu, lambda);
            error = cv::norm(matrix - low_rank - sparse, cv::NORM_INF) / cv::norm(matrix, cv::NORM_L2);
        }

        logConvergence(error, iteration);
        return {low_rank, sparse};
    }

    [[nodiscard]] auto decomposeMultiChannel(const cv::Mat &matrix) const -> std::tuple<cv::Mat, cv::Mat> {
        std::vector<cv::Mat> channels, low_rank_channels, sparse_channels;
        cv::split(matrix, channels);

        for (auto &channel: channels)
            std::tie(low_rank_channels.emplace_back(), sparse_channels.emplace_back()) =
                    decomposeSingleChannel(channel);

        cv::Mat low_rank, sparse;
        cv::merge(low_rank_channels, low_rank);
        cv::merge(sparse_channels, sparse);
        return {low_rank, sparse};
    }

    [[nodiscard]] auto iterate(const cv::Mat &matrix, const cv::Mat &low_rank, const cv::Mat &sparse, cv::Mat &dual,
                               T &mu, T lambda) const -> std::tuple<cv::Mat, cv::Mat> {
        auto updated_low_rank = singularValueThreshold(matrix - sparse + dual / mu, T(1) / mu);
        auto updated_sparse = shrink(matrix - updated_low_rank + dual / mu, lambda / mu);
        dual += mu * (matrix - updated_low_rank - updated_sparse);
        mu = std::min(mu * T(1.5), T(1e6));
        return {updated_low_rank, updated_sparse};
    }

    [[nodiscard]] static auto singularValueThreshold(const cv::Mat &matrix, T tau) noexcept -> cv::Mat {
        cv::Mat s, u, vt;
        cv::SVDecomp(matrix, s, u, vt);
        return u * cv::Mat::diag(shrink(s, tau)) * vt;
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

    void logConvergence(T error, int iteration) const noexcept {
        (error <= tolerance_)
                ? LOG_INFO("Converged in {} iterations.", iteration)
                : LOG_WARN("Reached max iterations ({}) without full convergence. Error: {}", max_iterations_, error);
    }

    const T tolerance_;
    const int max_iterations_;
};

#endif // RPCA_HPP
