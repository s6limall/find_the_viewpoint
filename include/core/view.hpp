//
// Created by ayush on 5/21/24.
//

#ifndef VIEW_HPP
#define VIEW_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "../../include/core/camera.hpp"


// Represents a camera view with a 6 degrees of freedom (6DoF) pose and viewpoint manipulation.
class View {
public:
    Camera camera;  // Camera for the view.

    View();  // Constructor.
    ~View();  // Destructor.

    // Computes the camera pose matrix from a position and the center of the object.
    void computePoseFromPositionAndObjectCenter(Eigen::Vector3f position, Eigen::Vector3f object_center);

    // Get the camera pose for rendering.
    Eigen::Matrix4f getCameraPose() const { return camera.getPose(); }
};

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
