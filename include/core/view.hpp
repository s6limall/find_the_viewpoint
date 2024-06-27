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
        View();

        ~View();

        // Computes the camera pose matrix from a position and the center of the object.
        void computePoseFromPositionAndObjectCenter(Eigen::Vector3f position, Eigen::Vector3f object_center);

        // Get the camera pose for rendering.
        Eigen::Matrix4f getPose() const;

        Eigen::VectorXd toVector() const;

    private:
        Eigen::Matrix4f pose_; // Pose of the view relative to the object
        std::shared_ptr<Camera> camera_; // Unique Camera for this view
    };
}

#endif // VIEW_HPP


/*// Represents a camera view with a 6 degrees of freedom (6DoF) pose and viewpoint manipulation.
class View {
public:
    Eigen::Matrix4d pose_6d;  // 4x4 transformation matrix representing the pose.

    View();
    ~View();

    // Computes the camera pose matrix from a position and the center of the object.
    void computePoseFromPositionAndObjectCenter(const Eigen::Vector3d& position, const Eigen::Vector3d& object_center);

    // Returns the pose as a 4x4 matrix of floats.
    Eigen::Matrix4f getCameraPose() const { return pose_6d.cast<float>(); }

    // Sets the camera's position in the world.
    void setPosition(float x, float y, float z);

    // Aligns the camera to face towards a specific target point in 3D space.
    void lookAt(const Eigen::Vector3d& object_center);
};

#endif // VIEW_HPP*/
