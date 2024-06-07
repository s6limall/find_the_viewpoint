//
// Created by ayush on 6/7/24.
//

#include "../../include/core/camera.hpp"

// Constructor initializes the camera position to the origin.
Camera::Camera() {
    pose.setIdentity();  // Initialize to the identity matrix.
}

// Destructor.
Camera::~Camera() {}

// Set the camera's position.
void Camera::setPosition(float x, float y, float z) {
    pose.setIdentity();  // Reset pose to identity.
    pose.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);  // Set translation part.
}

// Orient the camera to look at a specific target in 3D space.
void Camera::lookAt(const Eigen::Vector3f &target_center) {
    Eigen::Vector3f forward = (target_center - pose.block<3, 1>(0, 3)).normalized();  // Compute forward direction.
    Eigen::Vector3f world_up(0, 1, 0);  // Global up direction.
    Eigen::Vector3f right = world_up.cross(forward).normalized();  // Compute right direction.
    Eigen::Vector3f up = forward.cross(right).normalized();  // Compute up direction.

    // Set the rotation matrix.
    pose.block<3, 3>(0, 0) << right, up, forward;
}
