/*

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "processing/image/comparison/ssim_comparator.hpp"

using namespace processing::image;
using ::testing::_;
using ::testing::Return;

// Mock class for cv::Mat
class MockCVImage {
public:
    MOCK_CONST_METHOD0(rows, int());
    MOCK_CONST_METHOD0(cols, int());
    MOCK_CONST_METHOD1(at, float&(const cv::Point&));
    // Define other necessary mock methods or data access as per SSIMComparator requirements
};

// Mock class for Eigen::MatrixXd
class MockEigenImage {
public:
    MOCK_CONST_METHOD0(rows, int());
    MOCK_CONST_METHOD0(cols, int());
    // Define other necessary mock methods or data access as per SSIMComparator requirements
};

// Fixture class for setting up SSIMComparator tests
template<typename ImageType>
class SSIMComparatorTest : public ::testing::Test {
protected:
    SSIMComparator<ImageType> comparator;
    MockCVImage mockCVImage1, mockCVImage2;
    MockEigenImage mockEigenImage1, mockEigenImage2;

    void SetUp() override {
        // Define default behavior for cv::Mat mock objects
        ON_CALL(mockCVImage1, rows()).WillByDefault(Return(100));
        ON_CALL(mockCVImage1, cols()).WillByDefault(Return(100));
        ON_CALL(mockCVImage1, at(_)).WillByDefault(Return(0.0f)); // Example, adjust as needed

        ON_CALL(mockCVImage2, rows()).WillByDefault(Return(100));
        ON_CALL(mockCVImage2, cols()).WillByDefault(Return(100));
        ON_CALL(mockCVImage2, at(_)).WillByDefault(Return(0.0f)); // Example, adjust as needed

        // Define default behavior for Eigen::MatrixXd mock objects
        ON_CALL(mockEigenImage1, rows()).WillByDefault(Return(100));
        ON_CALL(mockEigenImage1, cols()).WillByDefault(Return(100));

        ON_CALL(mockEigenImage2, rows()).WillByDefault(Return(100));
        ON_CALL(mockEigenImage2, cols()).WillByDefault(Return(100));
    }
};

// Test case for cv::Mat
TYPED_TEST_SUITE_P(SSIMComparatorTest);

TYPED_TEST_P(SSIMComparatorTest, CompareImages) {
    // Set up expectations and actions for cv::Mat
    EXPECT_CALL(this->mockCVImage1, rows()).Times(::testing::AtLeast(1));
    EXPECT_CALL(this->mockCVImage1, cols()).Times(::testing::AtLeast(1));
    EXPECT_CALL(this->mockCVImage1, at(_)).Times(::testing::AtLeast(1));

    EXPECT_CALL(this->mockCVImage2, rows()).Times(::testing::AtLeast(1));
    EXPECT_CALL(this->mockCVImage2, cols()).Times(::testing::AtLeast(1));
    EXPECT_CALL(this->mockCVImage2, at(_)).Times(::testing::AtLeast(1));

    // Perform SSIM comparison with cv::Mat mock objects
    double ssimScoreCV = this->comparator.compare(this->mockCVImage1, this->mockCVImage2);

    // Add more assertions based on expected behavior
    EXPECT_GE(ssimScoreCV, 0.0); // Ensure score is non-negative, etc.

    // Set up expectations and actions for Eigen::MatrixXd
    EXPECT_CALL(this->mockEigenImage1, rows()).Times(::testing::AtLeast(1));
    EXPECT_CALL(this->mockEigenImage1, cols()).Times(::testing::AtLeast(1));

    EXPECT_CALL(this->mockEigenImage2, rows()).Times(::testing::AtLeast(1));
    EXPECT_CALL(this->mockEigenImage2, cols()).Times(::testing::AtLeast(1));

    // Perform SSIM comparison with Eigen::MatrixXd mock objects
    double ssimScoreEigen = this->comparator.compare(this->mockEigenImage1, this->mockEigenImage2);

    // Add more assertions based on expected behavior
    EXPECT_GE(ssimScoreEigen, 0.0); // Ensure score is non-negative, etc.
}

// Register the test cases with different ImageTypes
REGISTER_TYPED_TEST_SUITE_P(SSIMComparatorTest, CompareImages);

// Instantiate and define the test cases for both cv::Mat and Eigen::MatrixXd
typedef ::testing::Types<cv::Mat, Eigen::MatrixXd> ImageTypes;
INSTANTIATE_TYPED_TEST_SUITE_P(SSIMComparatorTests, SSIMComparatorTest, ImageTypes);


 */