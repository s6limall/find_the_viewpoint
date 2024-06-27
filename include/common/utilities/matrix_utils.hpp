// File: common/utilities/matrix_utils.hpp

#ifndef MATRIX_UTILS_HPP
#define MATRIX_UTILS_HPP

#include <opencv2/core.hpp>
#include <Eigen/Core>

namespace common::utilities {

    /**
     * @brief Converts an Eigen matrix to an OpenCV matrix.
     *
     * @tparam Derived Eigen matrix type.
     * @param eigen_matrix The input Eigen matrix.
     * @return The output OpenCV matrix.
     */
    template <typename Derived>
    cv::Mat eigenToCvMat(const Eigen::MatrixBase<Derived>& eigen_matrix) {
        cv::Mat cv_matrix(eigen_matrix.rows(), eigen_matrix.cols(), CV_64F);
        for (int i = 0; i < eigen_matrix.rows(); ++i) {
            for (int j = 0; j < eigen_matrix.cols(); ++j) {
                cv_matrix.at<double>(i, j) = eigen_matrix(i, j);
            }
        }
        return cv_matrix;
    }

    /**
     * @brief Converts an OpenCV matrix to an Eigen matrix.
     *
     * @tparam Derived Eigen matrix type.
     * @param cv_matrix The input OpenCV matrix.
     * @return The output Eigen matrix.
     */
    template <typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> cvMatToEigen(const cv::Mat& cv_matrix) {
        Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> eigen_matrix(cv_matrix.rows, cv_matrix.cols);
        for (int i = 0; i < cv_matrix.rows; ++i) {
            for (int j = 0; j < cv_matrix.cols; ++j) {
                eigen_matrix(i, j) = cv_matrix.at<double>(i, j);
            }
        }
        return eigen_matrix;
    }

}

#endif //MATRIX_UTILS_HPP
