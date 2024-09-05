// File: core/view.hpp

#ifndef VIEW_HPP
#define VIEW_HPP

#include <memory>
#include <Eigen/Core>

#include "core/camera.hpp"

namespace core {
    // Represents a camera view with a 6 degrees of freedom (6DoF) pose and viewpoint manipulation.
    class View {
    public:
        View() noexcept;

        // Computes the camera pose matrix from a position and the center of the object.
        void computePose(const Eigen::Vector3d &position, const Eigen::Vector3d &object_center);

        // Get the camera pose for rendering.
        [[nodiscard]] Eigen::Matrix4d getPose() const noexcept;

        // Set the camera extrinsics.
        void setPose(const Camera::Extrinsics &extrinsics) const noexcept;

        // Get the camera position.
        [[nodiscard]] Eigen::Vector3d getPosition() const noexcept;

        // Change the camera position.
        void setPosition(const Eigen::Vector3d &position) const noexcept;

        // Get the object center.
        [[nodiscard]] Eigen::Vector3d getObjectCenter() const noexcept;

        // Set position of the object center.
        void setObjectCenter(const Eigen::Vector3d &object_center) const noexcept;

        // Creates a view from a position.
        static View fromPosition(const Eigen::Vector3d &position,
                                 const Eigen::Vector3d &object_center = Eigen::Vector3d::Zero());

    private:
        Eigen::Matrix4d pose_; // Pose of the view relative to the object
        std::shared_ptr<Camera> camera_;
    };
}

#endif // VIEW_HPP
