// File: core/camera.hpp

#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>


namespace core {
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

    class Camera {
    public:
        struct CameraParameters {
            int width, height; // Image dimensions
            float fov_x, fov_y; // Horizontal and vertical field of view in radians
            Eigen::Matrix3f intrinsics; // Intrinsic matrix of the camera

            [[nodiscard]] float getFocalLengthX() const;

            [[nodiscard]] float getFocalLengthY() const;

            [[nodiscard]] float getPrincipalPointX() const;

            [[nodiscard]] float getPrincipalPointY() const;
        };

        Camera();

        // Get camera parameters (intrinsics, width, height, fov)
        [[nodiscard]] CameraParameters getParameters() const;

        void setIntrinsics(int width, int height, float fov_x, float fov_y);

        [[nodiscard]] Eigen::Matrix3f getIntrinsics() const;

        [[nodiscard]] Eigen::Matrix4f getPose() const; // Returns the current camera pose as a 4x4 matrix of floats.

        void setPose(const Eigen::Matrix4f &pose);

        // Get and set camera position (translation part)
        [[nodiscard]] Eigen::Vector3f getPosition() const;

        void setPosition(float x, float y, float z);

        [[nodiscard]] Eigen::Vector3f getObjectCenter() const;


        void lookAt(const Eigen::Vector3f &target_center); // face towards target point/center in 3D space.

    private:
        CameraParameters parameters_; // Camera parameters (intrinsics, width, height, fov)
        Eigen::Matrix4f pose_; // Camera pose (extrinsics)
        Eigen::Vector3f object_center_; // Object/target center that the camera is looking at

    };
}

#endif // CAMERA_HPP
