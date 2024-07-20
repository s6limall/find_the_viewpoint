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
        struct Intrinsics {
            int width{}, height{}; // Image dimensions
            double fov_x{}, fov_y{}; // Field of view in radians
            Eigen::Matrix3d matrix{Eigen::Matrix3d::Identity()}; // Intrinsic matrix

            constexpr Intrinsics() noexcept = default;

            [[nodiscard]] double getFocalLengthX() const noexcept { return matrix(0, 0); }
            [[nodiscard]] double getFocalLengthY() const noexcept { return matrix(1, 1); }
            [[nodiscard]] double getPrincipalPointX() const noexcept { return matrix(0, 2); }
            [[nodiscard]] double getPrincipalPointY() const noexcept { return matrix(1, 2); }
            [[nodiscard]] Eigen::Matrix3d getMatrix() const noexcept { return matrix; }


            void setIntrinsics(int width, int height, double fov_x, double fov_y);
        };

        struct Extrinsics {
            Eigen::Matrix4d matrix{Eigen::Matrix4d::Identity()}; // Extrinsic matrix (pose)

            constexpr Extrinsics() = default;

            [[nodiscard]] Eigen::Vector3d getTranslation() const noexcept { return matrix.block<3, 1>(0, 3); }
            [[nodiscard]] Eigen::Matrix3d getOrientation() const noexcept { return matrix.block<3, 3>(0, 0); }
            [[nodiscard]] Eigen::Matrix4d getMatrix() const noexcept { return matrix; }

            void setTranslation(double x, double y, double z) noexcept;

            void setOrientation(const Eigen::Matrix3d &orientation) noexcept;

            void setPose(const Eigen::Matrix4d &pose) noexcept;

            static Extrinsics fromPose(const Eigen::Matrix4d &pose) noexcept;
        };

        constexpr Camera() noexcept = default;

        [[nodiscard]] Intrinsics getIntrinsics() const noexcept;

        void setIntrinsics(int width, int height, double fov_x, double fov_y);

        [[nodiscard]] Extrinsics getExtrinsics() const noexcept;

        void setExtrinsics(const Extrinsics &extrinsics) noexcept;

        [[nodiscard]] Eigen::Vector3d getPosition() const noexcept; // camera position (translation part)

        Camera &setPosition(double x, double y, double z) noexcept;

        [[nodiscard]] Eigen::Matrix3d getRotation() const noexcept;

        Camera &setRotation(const Eigen::Matrix3d &rotation) noexcept;

        [[nodiscard]] Eigen::Vector3d getObjectCenter() const noexcept;

        Camera &lookAt(const Eigen::Vector3d &target_center) noexcept; // face towards target point/center in 3D space.

        // New methods for projecting points and computing Jacobian
        [[nodiscard]] Eigen::Vector2d project(const Eigen::Vector3d &point) const noexcept;

        [[nodiscard]] Eigen::Matrix<double, 2, 3> projectJacobian(const Eigen::Vector3d &point) const noexcept;

    private:
        Intrinsics intrinsics_; // Camera intrinsics (width, height, fov_x, fov_y, intrinsic matrix)
        Extrinsics extrinsics_; // Camera extrinsics (pose matrix)
        Eigen::Vector3d object_center_{Eigen::Vector3d::Zero()}; // Object/target center that the camera is looking at
    };
} // namespace core

#endif // CAMERA_HPP
