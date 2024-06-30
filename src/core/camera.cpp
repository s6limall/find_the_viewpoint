// File: core/camera.cpp

#include "core/camera.hpp"

#include <cmath>

#include "common/logging/logger.hpp"
#include "common/utilities/camera_utils.hpp"

namespace core {

    Camera::Camera() {
        parameters_.intrinsics.setIdentity();
        pose_.setIdentity();
        object_center_ = Eigen::Vector3f::Zero();
        LOG_INFO("Camera initialized with identity pose and intrinsics.");
    }

    Eigen::Vector3f Camera::getPosition() const {
        return pose_.block<3, 1>(0, 3);
    }

    void Camera::setPosition(float x, float y, float z) {
        pose_.setIdentity(); // Reset pose to identity.
        pose_.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z); // Set translation part.
        LOG_DEBUG("Camera position set to: ({}, {}, {})", x, y, z);
    }

    void Camera::lookAt(const Eigen::Vector3f &target_center) {
        object_center_ = target_center;
        const Eigen::Vector3f forward = (object_center_ - pose_.block<3, 1>(0, 3)).normalized();
        const Eigen::Vector3f world_up(0, 1, 0); // Global up direction.
        const Eigen::Vector3f right = world_up.cross(forward).normalized(); // Compute right direction.
        const Eigen::Vector3f up = forward.cross(right).normalized(); // Compute up direction.

        pose_.block<3, 3>(0, 0) << right, up, forward;

        LOG_DEBUG("Camera orientation set to look at: ({}, {}, {})", target_center.x(), target_center.y(),
                  target_center.z());
    }

    Eigen::Vector3f Camera::getObjectCenter() const {
        return object_center_;
    }

    void Camera::setIntrinsics(const int width, const int height, const float fov_x, const float fov_y) {
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("Width and height must be positive integers.");
        }

        const float fov_x_rad = common::utilities::toRadiansIfDegrees(fov_x);
        const float fov_y_rad = common::utilities::toRadiansIfDegrees(fov_y);

        LOG_DEBUG("Calculating focal lengths from width={}, height={}, fov_x={}, fov_y={}", width, height, fov_x,
                  fov_y);

        const float fx = common::utilities::calculateFocalLength(static_cast<float>(width), fov_x_rad);
        const float fy = common::utilities::calculateFocalLength(static_cast<float>(height), fov_y_rad);

        LOG_DEBUG("Calculated focal lengths: fx={}, fy={}", fx, fy);

        parameters_.intrinsics << fx, 0, static_cast<float>(width) / 2,
                0, fy, static_cast<float>(height) / 2,
                0, 0, 1;

        LOG_DEBUG("Intrinsic matrix set: {}", parameters_.intrinsics);

        parameters_.width = width;
        parameters_.height = height;
        parameters_.fov_x = fov_x_rad;
        parameters_.fov_y = fov_y_rad;

        LOG_INFO("Camera intrinsics set: fx={}, fy={}, cx={}, cy={}", fx, fy, static_cast<float>(width) / 2,
                 static_cast<float>(height) / 2);
    }

    Eigen::Matrix3f Camera::getIntrinsics() const {
        return parameters_.intrinsics;
    }

    Eigen::Matrix4f Camera::getPose() const {
        return pose_;
    }

    void Camera::setPose(const Eigen::Matrix4f &pose) {
        LOG_INFO("Setting camera pose to: {}", pose);
        pose_ = pose;
    }

    Camera::CameraParameters Camera::getParameters() const {
        return parameters_;
    }

    // Struct function definitions
    float Camera::CameraParameters::getFocalLengthX() const {
        return intrinsics(0, 0);
    }

    float Camera::CameraParameters::getFocalLengthY() const {
        return intrinsics(1, 1);
    }

    float Camera::CameraParameters::getPrincipalPointX() const {
        return intrinsics(0, 2);
    }

    float Camera::CameraParameters::getPrincipalPointY() const {
        return intrinsics(1, 2);
    }
}
