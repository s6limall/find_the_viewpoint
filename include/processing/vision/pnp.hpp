// File: processing/viewpoint/pnp.hpp

#ifndef PROCESSING_PNP_HPP
#define PROCESSING_PNP_HPP

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include "common/utilities/matrix.hpp"
#include "core/camera.hpp"

namespace processing::viewpoint {
    template<typename Scalar = double>
    class PnPSolver {
    public:
        struct Config {
            int ransac_iterations = 10; // Reduced number of iterations
            double reprojection_error = 8.0;
            double confidence = 0.99;
            cv::SolvePnPMethod pnp_method = cv::SOLVEPNP_EPNP;
        };

        explicit PnPSolver(const core::Camera::Intrinsics &intrinsics, const Config &config = Config()) :
            intrinsics_(intrinsics), config_(config) {}

        std::optional<std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>>
        estimatePose(const std::vector<Eigen::Matrix<Scalar, 3, 1>> &object_points,
                     const std::vector<Eigen::Matrix<Scalar, 2, 1>> &image_points) const {

            if (object_points.size() < 4 || image_points.size() < 4) {
                LOG_ERROR("Not enough points for PnP. Need at least 4 points.");
                return std::nullopt;
            }


            // TODO: USE .toView()
            std::vector<cv::Point3f> cv_object_points;
            std::vector<cv::Point2f> cv_image_points;

            for (const auto &pt: object_points) {
                cv_object_points.emplace_back(pt.x(), pt.y(), pt.z());
            }

            LOG_DEBUG("Object points: {}", cv_object_points.size());

            for (const auto &pt: image_points) {
                cv_image_points.emplace_back(pt.x(), pt.y());
            }

            LOG_DEBUG("Image points: {}", cv_image_points.size());

            cv::Mat rvec, tvec, rotation_matrix;
            std::vector<int> inliers;

            LOG_DEBUG("Solving PnP using method {}.", static_cast<int>(config_.pnp_method));

            cv::Mat camera_matrix;
            cv::eigen2cv(intrinsics_.getMatrix(), camera_matrix);

            // Create distortion coefficients (assume no distortion for now)
            cv::Mat dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);

            bool success = cv::solvePnPRansac(cv_object_points, cv_image_points, camera_matrix, dist_coeffs, rvec, tvec,
                                              false, config_.ransac_iterations, config_.reprojection_error,
                                              config_.confidence, inliers, config_.pnp_method);

            LOG_DEBUG("PnP solution found: {}", success);

            if (success) {
                LOG_DEBUG("PnP solution found with {} inliers.", inliers.size());
                Eigen::Matrix<Scalar, 3, 1> translation;
                Eigen::Matrix<Scalar, 3, 3> rotation;
                LOG_DEBUG("Converting translation vector.");
                cv::cv2eigen(tvec, translation);
                LOG_DEBUG("Converting rotation vector to rotation matrix.");
                cv::Rodrigues(rvec, rotation_matrix); // Convert rotation vector to rotation matrix
                cv::cv2eigen(rotation_matrix, rotation);

                return std::make_optional(std::make_pair(translation, Eigen::Quaternion<Scalar>(rotation)));
            } else {
                LOG_ERROR("solvePnPRansac failed.");
                LOG_ERROR("Number of inliers: {}", inliers.size());
            }
            return std::nullopt;
        }

    private:
        const core::Camera::Intrinsics &intrinsics_;
        Config config_;
    };
} // namespace processing::viewpoint

#endif // PROCESSING_PNP_HPP
