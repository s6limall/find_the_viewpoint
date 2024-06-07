//
// Created by ayush on 6/6/24.
//

#include "../../include/core/camera_viewpoint.hpp"

// TODO: Delete. Made redundant by Camera and View.

// Initialize the camera at a specific position.
CameraViewpoint::CameraViewpoint(float x, float y, float z) {
    setPosition(x, y, z);  // Set initial camera position.
}

// Sets the position of the camera in the 3D space.
void CameraViewpoint::setPosition(float x, float y, float z) {
    pose.setIdentity();  // Reset pose to identity to avoid cumulative transformations.
    pose.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);  // Set the translation part of the pose matrix.
}

// Orients the camera to look at a specified point in space.
void CameraViewpoint::lookAt(const Eigen::Vector3f &target_center) {
    Eigen::Vector3f forward = (target_center - pose.block<3, 1>(0, 3)).normalized();  // Calculate forward direction.
    Eigen::Vector3f world_up(0, 1, 0);  // World's up direction, usually the y-axis.
    Eigen::Vector3f right = world_up.cross(forward).normalized();  // Calculate right vector orthogonal to up and forward.
    Eigen::Vector3f up = forward.cross(right).normalized();  // Recalculate up vector for a stable orientation.

    // Set the rotation part of the pose matrix.
    pose.block<3, 3>(0, 0) << right, up, forward;
}


/*
void CameraViewpoint::randomizePosition(double radius) {
    float theta = static_cast<float>(rand()) / RAND_MAX * 2 * M_PI;
    float phi = static_cast<float>(rand()) / RAND_MAX * M_PI - M_PI / 2;
    float x = radius * cos(phi) * cos(theta);
    float y = radius * cos(phi) * sin(theta);
    float z = radius * sin(phi);
    pose.block<3, 1>(0, 3) = Eigen::Vector3f(x, y, z);
}

void CameraViewpoint::adjustPose(const Eigen::Vector3f &adjustment) {
    // Extract roll, pitch, and yaw from the adjustment vector
    float roll = adjustment[0];
    float pitch = adjustment[1];
    float yaw = adjustment[2];

    // Calculate rotation matrices around the X, Y, and Z axes
    Eigen::Matrix3f rollMatrix;
    rollMatrix = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());

    Eigen::Matrix3f pitchMatrix;
    pitchMatrix = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());

    Eigen::Matrix3f yawMatrix;
    yawMatrix = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());

    // Combine the rotations
    Eigen::Matrix3f rotation = yawMatrix * pitchMatrix * rollMatrix;

    // Update the pose matrix
    pose.block<3, 3>(0, 0) = rotation * pose.block<3, 3>(0, 0);
}
*/
