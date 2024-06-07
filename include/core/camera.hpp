//
// Created by ayush on 6/7/24.
//

#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

/**
 * @brief Manages the camera's position and orientation using yaw, pitch, and roll.
 *
 * This class provides functionality to manage the camera's pose in a 3D space with 6 Degrees of Freedom (6DoF).
 * The camera pose is represented by a 4x4 transformation matrix, where:
 * - The first 3 columns represent the rotation matrix.
 * - The last column represents the translation vector.
 *
 * The camera pose can be adjusted by changing the yaw, pitch, and roll angles:
 * - Yaw: Rotation about the vertical axis (up), changes the direction the camera is facing left or right.
 * - Pitch: Rotation about the side-to-side axis (right), changes the direction the camera is facing up or down.
 * - Roll: Rotation about the axis pointing forward, tilts the camera side to side.
 */

// Class for managing camera pose and transformations.
class Camera {
public:
    Eigen::Matrix4f pose;  // Transformation matrix for the camera.

    Camera();
    ~Camera();

    // Set the camera position in 3D space.
    void setPosition(float x, float y, float z);

    // Aligns the camera to face towards a specific target point/center in 3D space.
    void lookAt(const Eigen::Vector3f &target_center);

    // Returns the current camera pose as a 4x4 matrix of floats.
    Eigen::Matrix4f getPose() const { return pose; }
};

#endif // CAMERA_HPP
