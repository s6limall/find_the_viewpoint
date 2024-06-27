// File: core/camera.cpp

#include "core/camera.hpp"

#include <spdlog/spdlog.h>
#include <cmath>

namespace core {

    Camera::Camera() {
        config_.intrinsics.setIdentity();
        pose_.setIdentity(); // Initialize to the identity matrix.
        spdlog::info("Camera initialized with identity pose and intrinsics.");
    }

    Camera::~Camera() = default;

    void Camera::setPosition(float x, float y, float z) {
        pose_.setIdentity(); // Reset pose to identity.
        pose_.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z); // Set translation part.
        spdlog::debug("Camera position set to: ({}, {}, {})", x, y, z);
    }

    void Camera::lookAt(const Eigen::Vector3f &target_center) {
        const Eigen::Vector3f forward = (target_center - pose_.block<3, 1>(0, 3)).normalized();
        const Eigen::Vector3f world_up(0, 1, 0); // Global up direction.
        const Eigen::Vector3f right = world_up.cross(forward).normalized(); // Compute right direction.
        const Eigen::Vector3f up = forward.cross(right).normalized(); // Compute up direction.

        pose_.block<3, 3>(0, 0) << right, up, forward;

        spdlog::debug("Camera orientation set to look at: ({}, {}, {})", target_center.x(), target_center.y(),
                      target_center.z());
    }

    void Camera::setIntrinsics(int width, int height, float fov_x, float fov_y) {
        float fov_x_rad = toRadians(fov_x);
        float fov_y_rad = toRadians(fov_y);

        float fx = calculateFocalLength(static_cast<float>(width), fov_x_rad);
        float fy = calculateFocalLength(static_cast<float>(height), fov_y_rad);

        config_.intrinsics << fx, 0, static_cast<float>(width) / 2,
                0, fy, static_cast<float>(height) / 2,
                0, 0, 1;

        config_.width = width;
        config_.height = height;
        config_.fov_x = fov_x_rad;
        config_.fov_y = fov_y_rad;

        spdlog::debug("Camera intrinsics set: fx={}, fy={}, cx={}, cy={}", fx, fy, static_cast<float>(width) / 2,
                      static_cast<float>(height) / 2);
    }

    Eigen::Matrix3f Camera::getIntrinsics() const {
        spdlog::debug("Returning camera intrinsics.");
        return config_.intrinsics;
    }

    Eigen::Matrix4f Camera::getPose() const {
        spdlog::debug("Returning camera pose.");
        return pose_;
    }

    void Camera::setPose(const Eigen::Matrix4f &pose) {
        pose_ = pose;
        spdlog::info("Camera pose set.");
    }

    std::pair<double, double> Camera::calculateDistanceBounds(double object_scale, double min_scale, double max_scale) {
        double min_distance = std::max(min_scale / object_scale, 0.1);
        double max_distance = std::min(max_scale / object_scale, 10.0);
        spdlog::debug("Calculated distance bounds: min_distance = {}, max_distance = {}", min_distance, max_distance);
        return {min_distance, max_distance};
    }

    Camera::CameraConfig Camera::getConfig() const {
        spdlog::debug("Returning camera configuration.");
        return config_;
    }

    float Camera::toRadians(float degrees) {
        if (std::abs(degrees) > 2 * M_PI) {
            spdlog::debug("Converting degrees to radians: {}", degrees);
            return degrees * static_cast<float>(M_PI) / 180.0f;
        }
        return degrees;
    }

    float Camera::calculateFocalLength(float size, float fov_rad) {
        float focal_length = size / (2.0f * std::tan(fov_rad / 2.0f));
        spdlog::debug("Calculated focal length: {} for size: {} and fov_rad: {}", focal_length, size, fov_rad);
        return focal_length;
    }
}
