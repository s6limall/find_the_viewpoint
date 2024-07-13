// File: common/utilities/matrix.hpp

#ifndef COMMON_UTILITIES_MATRIX_HPP
#define COMMON_UTILITIES_MATRIX_HPP

#include <opencv2/core.hpp>
#include <Eigen/Core>

#include "common/logging/logger.hpp"

namespace common::utilities {

    /**
     * @brief Converts an Eigen matrix to an OpenCV matrix.
     *
     * @tparam Derived Eigen matrix type.
     * @param eigen_matrix The input Eigen matrix.
     * @return The output OpenCV matrix.
     */
    template<typename Derived>
    cv::Mat toCV(const Eigen::MatrixBase<Derived> &eigen_matrix) noexcept {
        using Scalar = typename Derived::Scalar;
        constexpr int type = std::is_same_v<Scalar, float> ? CV_32F : CV_64F;

        LOG_DEBUG("Converting Eigen matrix to OpenCV matrix (size: {}x{})", eigen_matrix.rows(), eigen_matrix.cols());

        cv::Mat cv_matrix(eigen_matrix.rows(), eigen_matrix.cols(), type);
        for (size_t i = 0; i < eigen_matrix.rows(); ++i) {
            for (size_t j = 0; j < eigen_matrix.cols(); ++j) {
                cv_matrix.at<Scalar>(i, j) = eigen_matrix(i, j);
            }
        }

        LOG_DEBUG("Conversion successful.");
        return cv_matrix;
    }

    /**
     * @brief Converts an OpenCV matrix to an Eigen matrix.
     *
     * @tparam Derived Eigen matrix type.
     * @param cv_matrix The input OpenCV matrix.
     * @return The output Eigen matrix.
     */
    template<typename Derived>
    auto toEigen(const cv::Mat &cv_matrix) {
        using Scalar = typename Derived::Scalar;
        constexpr int type = std::is_same_v<Scalar, float> ? CV_32F : CV_64F;

        LOG_DEBUG("Converting OpenCV matrix to Eigen matrix (size: {}x{})", cv_matrix.rows, cv_matrix.cols);

        if (cv_matrix.type() != type) {
            LOG_ERROR("cv::Mat type does not match Eigen matrix scalar type.");
            throw std::invalid_argument("cv::Mat type does not match Eigen matrix scalar type.");
        }

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix(cv_matrix.rows, cv_matrix.cols);
        for (size_t i = 0; i < cv_matrix.rows; ++i) {
            for (size_t j = 0; j < cv_matrix.cols; ++j) {
                eigen_matrix(i, j) = cv_matrix.at<Scalar>(i, j);
            }
        }

        LOG_DEBUG("Conversion successful.");
        return eigen_matrix;
    }

}

#endif // COMMON_UTILITIES_MATRIX_HPP
