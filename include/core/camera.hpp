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
        struct CameraConfig {
            int width, height; // Image dimensions
            float fov_x, fov_y; // Horizontal and vertical field of view in radians
            Eigen::Matrix3f intrinsics; // Intrinsic matrix of the camera
        };

        Camera();

        ~Camera();

        // Set camera intrinsics
        void setIntrinsics(int width, int height, float fov_x, float fov_y);

        [[nodiscard]] Eigen::Matrix3f getIntrinsics() const;

        // Set and get camera pose
        void setPosition(float x, float y, float z); // Set the camera position in 3D space.
        void lookAt(const Eigen::Vector3f &target_center); // face towards target point/center in 3D space.

        [[nodiscard]] Eigen::Matrix4f getPose() const; // Returns the current camera pose as a 4x4 matrix of floats.

        void setPose(const Eigen::Matrix4f &pose);

        // Get configuration
        [[nodiscard]] CameraConfig getConfig() const;

        // Calculate distance bounds based on object scale
        static std::pair<double, double> calculateDistanceBounds(double object_scale, double min_scale = 0.05,
                                                                 double max_scale = 0.5);

    private:
        CameraConfig config_;
        Eigen::Matrix4f pose_; // Camera pose (extrinsics)

        // Helper function to convert degrees to radians
        static float toRadians(float degrees);

        // Calculate the focal length based on the field of view
        static float calculateFocalLength(float size, float fov_rad);
    };
}

#endif // CAMERA_HPP
