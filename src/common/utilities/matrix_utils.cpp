// File: common/utilities/matrix_utils.cpp

#include "common/utilities/matrix_utils.hpp"

namespace common::utilities {

    void convertEigenToCvMat(const Eigen::Matrix4f &eigen_matrix, cv::Mat &rotation_matrix,
                                 cv::Mat &translation_vector) {
        // Ensure the input matrix is the correct size
        if (eigen_matrix.rows() != 4 || eigen_matrix.cols() != 4) {
            throw std::invalid_argument("Input matrix must be a 4x4 Eigen::Matrix4f.");
        }

        // Convert rotation and translation parts
        rotation_matrix = cv::Mat(3, 3, CV_64F);
        translation_vector = cv::Mat(3, 1, CV_64F);

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                rotation_matrix.at<double>(i, j) = eigen_matrix(i, j);
            }
            translation_vector.at<double>(i) = eigen_matrix(i, 3); // Copy the translation components
        }
    }

    void convertCvMatToEigen(const cv::Mat &rotation_matrix, const cv::Mat &translation_vector,
                             Eigen::Matrix4f &eigen_matrix) {
        // Ensure the input matrices are the correct size
        if (rotation_matrix.rows != 3 || rotation_matrix.cols != 3 || translation_vector.rows != 3 || translation_vector
            .cols != 1) {
            throw std::invalid_argument("Invalid matrix sizes for rotation or translation.");
            }

        // Initialize the Eigen matrix
        eigen_matrix.setIdentity();

        // Convert rotation and translation parts
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                eigen_matrix(i, j) = static_cast<float>(rotation_matrix.at<double>(i, j));
            }
            eigen_matrix(i, 3) = static_cast<float>(translation_vector.at<double>(i));
        }
    }
}