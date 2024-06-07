//
// Created by ayush on 6/6/24.
//

#ifndef CAMERA_VIEWPOINT_HPP
#define CAMERA_VIEWPOINT_HPP

#include <Eigen/Geometry>

// TODO: Delete. Made redundant by Camera and View.


// Manages the camera's position and orientation using yaw, pitch, and roll:
// (with 6 Degrees of Freedom (6DoF) for the camera pose).
// The camera pose is represented by a 4x4 transformation matrix.
// The first 3 columns of the matrix represent the rotation matrix, and the last column represents the translation vector.
// The camera pose can be adjusted by changing the yaw, pitch, and roll angles.
// Yaw: Rotation about the vertical axis (up), changes the direction the camera is facing left or right.
// Pitch: Rotation about the side-to-side axis (right), changes the direction the camera is facing up or down.
// Roll: Rotation about the axis pointing forward, tilts the camera side to side.

// Manages camera viewpoints .
class CameraViewpoint {
public:
    Eigen::Matrix4f pose;  // Represents the transformation matrix for the camera pose.

    // Constructor initializes camera position at (x, y, z).
    CameraViewpoint(float x, float y, float z);

    // Sets the camera position in the world space.
    void setPosition(float x, float y, float z);

    // Aligns the camera to look towards an object center from its current position.
    void lookAt(const Eigen::Vector3f &object_center);
};

#endif // CAMERA_VIEWPOINT_HPP
