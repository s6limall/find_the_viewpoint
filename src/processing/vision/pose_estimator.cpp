// File: processing/vision/pose_estimator.cpp

#include "processing/vision/pose_estimator.hpp"

#include "common/logging/logger.hpp"

namespace processing::vision {

    PoseEstimator::PoseEstimator(std::shared_ptr<core::Camera> camera) :
        camera_(std::move(camera)), pose_matrix_(Eigen::Matrix4d::Identity()) {
    }

    std::vector<cv::Point2f> PoseEstimator::detectKeypoints(const cv::Mat &image) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        orb->detect(image, keypoints);

        std::vector<cv::Point2f> points;
        cv::KeyPoint::convert(keypoints, points);

        return points;
    }

    std::vector<Eigen::Vector3d> PoseEstimator::generateObjectPoints(int rows, int cols, float square_size) {
        std::vector<Eigen::Vector3d> object_points;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                object_points.emplace_back(j * square_size, i * square_size, 0.0);
            }
        }
        return object_points;
    }

    Eigen::Matrix4d PoseEstimator::estimatePose(const cv::Mat &image) {
        std::vector<cv::Point2f> image_points = detectKeypoints(image);

        int rows = 6; // Number of rows in the grid
        int cols = 9; // Number of columns in the grid
        float square_size = 0.025f; // Size of a square in the grid in meters

        std::vector<Eigen::Vector3d> object_points = generateObjectPoints(rows, cols, square_size);

        if (image_points.size() < object_points.size()) {
            throw std::runtime_error("Not enough keypoints detected in the image.");
        }

        cv::Mat camera_matrix, dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
        cv::eigen2cv(camera_->getIntrinsics().getMatrix(), camera_matrix);

        cv::Mat rvec, tvec;
        std::vector<cv::Point3f> cv_object_points;
        for (const auto &point: object_points) {
            cv_object_points.emplace_back(point.x(), point.y(), point.z());
        }

        std::vector<cv::Point2f> selected_image_points(image_points.begin(),
                                                       image_points.begin() + object_points.size());

        cv::solvePnP(cv_object_points, selected_image_points, camera_matrix, dist_coeffs, rvec, tvec);

        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);

        Eigen::Matrix3d rotation;
        Eigen::Vector3d translation;

        cv::cv2eigen(rotation_matrix, rotation);
        cv::cv2eigen(tvec, translation);

        pose_matrix_.block<3, 3>(0, 0) = rotation;
        pose_matrix_.block<3, 1>(0, 3) = translation;

        return pose_matrix_;
    }

    Eigen::Matrix4d PoseEstimator::getPose() const noexcept {
        return pose_matrix_;
    }

}
