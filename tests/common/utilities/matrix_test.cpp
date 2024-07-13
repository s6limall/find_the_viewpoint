// File: tests/common/utilities/matrix_test.cpp

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "common/utilities/matrix.hpp"

// Test successful conversion from Eigen to OpenCV and back
TEST(MatrixUtilsTest, toCVAndBack) {
    Eigen::MatrixXd eigen_matrix(3, 3);
    eigen_matrix << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;

    const cv::Mat cv_matrix = common::utilities::toCV(eigen_matrix);
    const auto result_matrix = common::utilities::toEigen<Eigen::MatrixXd>(cv_matrix);

    EXPECT_EQ(eigen_matrix.rows(), cv_matrix.rows);
    EXPECT_EQ(eigen_matrix.cols(), cv_matrix.cols);
    EXPECT_EQ(cv_matrix.type(), CV_64F);
    EXPECT_EQ(eigen_matrix, result_matrix);
}

// Test conversion of different sizes
TEST(MatrixUtilsTest, DifferentSizes) {
    Eigen::MatrixXd eigen_matrix(2, 5);
    eigen_matrix << 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10;

    const cv::Mat cv_matrix = common::utilities::toCV(eigen_matrix);
    const auto result_matrix = common::utilities::toEigen<Eigen::MatrixXd>(cv_matrix);

    EXPECT_EQ(eigen_matrix.rows(), cv_matrix.rows);
    EXPECT_EQ(eigen_matrix.cols(), cv_matrix.cols);
    EXPECT_EQ(cv_matrix.type(), CV_64F);
    EXPECT_EQ(eigen_matrix, result_matrix);
}

// Test conversion with different scalar types
TEST(MatrixUtilsTest, DifferentScalarTypes) {
    Eigen::MatrixXf eigen_matrix(2, 2);
    eigen_matrix << 1.1f, 2.2f,
            3.3f, 4.4f;

    const cv::Mat cv_matrix = common::utilities::toCV(eigen_matrix);
    const auto result_matrix = common::utilities::toEigen<Eigen::MatrixXf>(cv_matrix);

    EXPECT_EQ(eigen_matrix.rows(), cv_matrix.rows);
    EXPECT_EQ(eigen_matrix.cols(), cv_matrix.cols);
    EXPECT_EQ(cv_matrix.type(), CV_32F);
    EXPECT_EQ(eigen_matrix, result_matrix);
}

// Test error handling for type mismatch
TEST(MatrixUtilsTest, TypeMismatch) {
    cv::Mat cv_matrix(2, 2, CV_32F);
    cv_matrix.at<float>(0, 0) = 1.0f;
    cv_matrix.at<float>(0, 1) = 2.0f;
    cv_matrix.at<float>(1, 0) = 3.0f;
    cv_matrix.at<float>(1, 1) = 4.0f;

    EXPECT_THROW({
                 try {
                 Eigen::MatrixXd result_matrix = common::utilities::toEigen<Eigen::MatrixXd>(cv_matrix);
                 } catch (const std::invalid_argument& e) {
                 EXPECT_STREQ("cv::Mat type does not match Eigen matrix scalar type.", e.what());
                 throw;
                 }
                 }, std::invalid_argument);
}
