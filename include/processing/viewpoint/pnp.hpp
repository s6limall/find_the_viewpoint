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
            int ransacIterations = 10; // Reduced number of iterations
            double reprojectionError = 8.0;
            double confidence = 0.99;
            cv::SolvePnPMethod pnpMethod = cv::SOLVEPNP_EPNP;
        };

        explicit PnPSolver(const core::Camera::Intrinsics &intrinsics, const Config &config = Config()) :
            intrinsics_(intrinsics), config_(config) {}

        std::optional<std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>>
        estimatePose(const std::vector<Eigen::Matrix<Scalar, 3, 1>> &objectPoints,
                     const std::vector<Eigen::Matrix<Scalar, 2, 1>> &imagePoints) const {

            if (objectPoints.size() < 4 || imagePoints.size() < 4) {
                LOG_ERROR("Not enough points for PnP. Need at least 4 points.");
                return std::nullopt;
            }


            // TODO: USE .toView()
            std::vector<cv::Point3f> cvObjectPoints;
            std::vector<cv::Point2f> cvImagePoints;

            for (const auto &pt: objectPoints) {
                cvObjectPoints.emplace_back(pt.x(), pt.y(), pt.z());
            }

            LOG_DEBUG("Object points: {}", cvObjectPoints.size());

            for (const auto &pt: imagePoints) {
                cvImagePoints.emplace_back(pt.x(), pt.y());
            }

            LOG_DEBUG("Image points: {}", cvImagePoints.size());

            cv::Mat rvec, tvec, rotMat;
            std::vector<int> inliers;

            LOG_DEBUG("Solving PnP using method {}.", static_cast<int>(config_.pnpMethod));

            cv::Mat camera_matrix;
            cv::eigen2cv(intrinsics_.getMatrix(), camera_matrix);

            // Create distortion coefficients (assume no distortion for now)
            cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);

            bool success = cv::solvePnPRansac(cvObjectPoints, cvImagePoints, camera_matrix, distCoeffs, rvec, tvec,
                                              false, config_.ransacIterations, config_.reprojectionError,
                                              config_.confidence, inliers, config_.pnpMethod);

            LOG_DEBUG("PnP solution found: {}", success);

            if (success) {
                LOG_DEBUG("PnP solution found with {} inliers.", inliers.size());
                Eigen::Matrix<Scalar, 3, 1> translation;
                Eigen::Matrix<Scalar, 3, 3> rotation;
                LOG_DEBUG("Converting translation vector.");
                cv::cv2eigen(tvec, translation);
                LOG_DEBUG("Converting rotation vector to rotation matrix.");
                cv::Rodrigues(rvec, rotMat); // Convert rotation vector to rotation matrix
                cv::cv2eigen(rotMat, rotation);

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
