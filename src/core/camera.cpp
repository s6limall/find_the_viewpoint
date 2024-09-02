// File: core/camera.cpp

#include "core/camera.hpp"

#include <cmath>

#include "common/logging/logger.hpp"
#include "common/utilities/camera.hpp"

namespace core {

    // Struct function definitions

    void Camera::Intrinsics::setIntrinsics(const int width, const int height, const double fov_x, const double fov_y) {
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("Width and height must be positive integers.");
        }

        const double fov_x_rad = common::utilities::toRadiansIfDegrees(fov_x);
        const double fov_y_rad = common::utilities::toRadiansIfDegrees(fov_y);

        const double fx = common::utilities::calculateFocalLength(static_cast<double>(width), fov_x_rad);
        const double fy = common::utilities::calculateFocalLength(static_cast<double>(height), fov_y_rad);
        const double cx = static_cast<double>(width) / 2;
        const double cy = static_cast<double>(height) / 2;

        LOG_TRACE("Calculated fx = {}, fy = {}, cx = {}, cy = {}.", fx, fy, cx, cy);

        this->matrix << fx, 0, cx, 0, fy, cy, 0, 0, 1;

        // LOG_DEBUG("Intrinsic matrix set: {}", this->matrix);

        this->width = width;
        this->height = height;
        this->fov_x = fov_x_rad;
        this->fov_y = fov_y_rad;

        LOG_INFO("Camera intrinsics set: fx={}, fy={}, cx={}, cy={}", fx, fy, cx, cy);
    }

    void Camera::Extrinsics::setTranslation(double x, double y, double z) noexcept {
        matrix.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z); // Set translation part.
        LOG_TRACE("Camera position set to: ({}, {}, {})", x, y, z);
    }

    void Camera::Extrinsics::setOrientation(const Eigen::Matrix3d &orientation) noexcept {
        matrix.block<3, 3>(0, 0) = orientation;
        LOG_TRACE("Camera rotation set.");
    }

    void Camera::Extrinsics::setPose(const Eigen::Matrix4d &pose) noexcept {
        matrix = pose;
        LOG_TRACE("Camera pose set.");
    }

    Camera::Extrinsics Camera::Extrinsics::fromPose(const Eigen::Matrix4d &pose) noexcept {
        Extrinsics extrinsics;
        extrinsics.setPose(pose);
        return extrinsics;
    }

    // Camera function definitions

    Camera::Intrinsics Camera::getIntrinsics() const noexcept { return intrinsics_; }

    void Camera::setIntrinsics(const int width, const int height, const double fov_x, const double fov_y) {
        intrinsics_.setIntrinsics(width, height, fov_x, fov_y);
    }

    Camera::Extrinsics Camera::getExtrinsics() const noexcept { return extrinsics_; }

    void Camera::setExtrinsics(const Extrinsics &extrinsics) noexcept { extrinsics_ = extrinsics; }

    Eigen::Vector3d Camera::getPosition() const noexcept { return extrinsics_.getTranslation(); }

    Camera &Camera::setPosition(const double x, const double y, const double z) noexcept {
        extrinsics_.matrix.setIdentity(); // Reset pose to identity.
        extrinsics_.setTranslation(x, y, z);
        return *this;
    }

    Eigen::Matrix3d Camera::getRotation() const noexcept { return extrinsics_.getOrientation(); }

    Camera &Camera::setRotation(const Eigen::Matrix3d &rotation) noexcept {
        extrinsics_.setOrientation(rotation);
        return *this;
    }

    Eigen::Vector3d Camera::getObjectCenter() const noexcept { return object_center_; }

    Camera &Camera::lookAt(const Eigen::Vector3d &target_center) noexcept {
        object_center_ = target_center;
        const Eigen::Vector3d forward = (object_center_ - extrinsics_.matrix.block<3, 1>(0, 3)).normalized();
        const Eigen::Vector3d world_up(0, 1, 0); // Global up direction.
        // const Eigen::Vector3d world_up(0, 0, 1); // Global up direction.
        const Eigen::Vector3d right = world_up.cross(forward).normalized(); // Compute right direction.
        const Eigen::Vector3d up = forward.cross(right).normalized(); // Compute up direction.

        extrinsics_.matrix.block<3, 3>(0, 0) << right, up, forward;

        LOG_TRACE("Camera orientation set to look at: ({}, {}, {})", target_center.x(), target_center.y(),
                  target_center.z());
        return *this;
    }

    // New methods for projecting points and computing Jacobian

    Eigen::Vector2d Camera::project(const Eigen::Vector3d &point) const noexcept {
        Eigen::Vector4d homogeneous_point(point.x(), point.y(), point.z(), 1.0);
        Eigen::Vector3d cam_point = intrinsics_.matrix * homogeneous_point.head<3>();
        return cam_point.hnormalized();
    }

    Eigen::Matrix<double, 2, 3> Camera::projectJacobian(const Eigen::Vector3d &point) const noexcept {
        const double fx = intrinsics_.getFocalLengthX();
        const double fy = intrinsics_.getFocalLengthY();
        const double inv_z = 1.0 / point.z();
        const double inv_z2 = inv_z * inv_z;

        Eigen::Matrix<double, 2, 3> J;
        J << fx * inv_z, 0, -fx * point.x() * inv_z2, 0, fy * inv_z, -fy * point.y() * inv_z2;
        return J;
    }

} // namespace core
