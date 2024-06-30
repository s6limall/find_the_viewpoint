// File: core/view.cpp

#include "core/view.hpp"

#include "common/logging/logger.hpp"

namespace core {
    View::View() :
        camera_(std::make_shared<Camera>()) {
        camera_->setPosition(0, 0, 0); // Initialize camera position to (0, 0, 0)
        pose_.setIdentity();
        LOG_DEBUG("View initialized. Camera position and pose set to identity.");
    }

    View::~View() = default;

    void View::computePoseFromPositionAndObjectCenter(Eigen::Vector3f position, Eigen::Vector3f object_center) {
        camera_->setPosition(position.x(), position.y(), position.z()); // Set camera position
        camera_->lookAt(object_center.cast<float>()); // Orient camera towards the object center
        pose_ = camera_->getPose(); // Update view pose with camera's current pose
        LOG_INFO("Computed view pose for position ({}, {}, {}) with object center ({}, {}, {})",
                 position.x(), position.y(), position.z(), object_center.x(), object_center.y(), object_center.z());
    }

    Eigen::Matrix4f View::getPose() const {
        return pose_;
    }

    Eigen::Vector3f View::getPosition() const {
        return camera_->getPosition();
    }

    Eigen::Vector3f View::getObjectCenter() const {
        return camera_->getObjectCenter();
    }

    Eigen::VectorXd View::toVector() const {
        Eigen::VectorXd vec(3);
        vec << pose_(0, 3), pose_(1, 3), pose_(2, 3);
        return vec;
    }


    /*void View::computePoseFromPositionAndObjectCenter(Eigen::Vector3d position, Eigen::Vector3d object_center) {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3,1>(0,3) = -position;

        Eigen::Vector3d Z = (object_center - position).normalized();
        Eigen::Vector3d X = (-Z).cross(Eigen::Vector3d(0, 1, 0)).normalized();
        Eigen::Vector3d Y = X.cross(-Z).normalized();

        Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
        R.block<3,3>(0,0) << X, Y, Z;

        pose_6d = (R.inverse() * T).inverse();
    }*/

}
