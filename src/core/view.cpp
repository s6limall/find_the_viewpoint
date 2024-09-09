// File: core/view.cpp

#include "core/view.hpp"

#include <config/configuration.hpp>

#include "common/logging/logger.hpp"

namespace core {
    View::View() noexcept:
        camera_(std::make_shared<Camera>()) {
        camera_->setPosition(0.0, 0.0, 0.0); // Initialize camera position to (0, 0, 0)
        pose_.setIdentity();
        LOG_TRACE("View initialized. Camera position and pose set to identity.");
    }

    void View::computePose(const Eigen::Vector3d &position,
                           const Eigen::Vector3d &object_center) {
        if (position.hasNaN() || object_center.hasNaN()) {
            throw std::invalid_argument("Position and object center must not contain NaN values.");
        }

        const auto x = config::get("origin.x", 0.0);
        const auto y = config::get("origin.y", 0.0);
        const auto z = config::get("origin.z", 0.0);

        const auto center = Eigen::Vector3d(x, y, z);

        camera_->setPosition(position.x(), position.y(), position.z()); // Set camera position
        camera_->lookAt(center); // Orient camera towards the object center
        pose_ = camera_->getExtrinsics().matrix; // Update view pose with camera's current pose
        LOG_TRACE("Computed view pose for position ({}, {}, {}) with object center ({}, {}, {})",
                  position.x(), position.y(), position.z(), center.x(), center.y(), center.z());
    }

    Eigen::Matrix4d View::getPose() const noexcept {
        return pose_;
    }

    void View::setPose(const Camera::Extrinsics &extrinsics) const noexcept {
        camera_->setExtrinsics(extrinsics);
    }


    Eigen::Vector3d View::getPosition() const noexcept {
        return camera_->getPosition();
    }

    void View::setPosition(const Eigen::Vector3d &position) const noexcept {
        camera_->setPosition(position.x(), position.y(), position.z());
    }

    Eigen::Vector3d View::getObjectCenter() const noexcept {
        return camera_->getObjectCenter();
    }

    void View::setObjectCenter(const Eigen::Vector3d &object_center) const noexcept {
        const auto x = config::get("origin.x", 0.0);
        const auto y = config::get("origin.y", 0.0);
        const auto z = config::get("origin.z", 0.0);

        const auto center = Eigen::Vector3d(x, y, z);
        camera_->lookAt(center);
    }

    View View::fromPosition(const Eigen::Vector3d &position, const Eigen::Vector3d &object_center) {
        View view;
        view.computePose(position, object_center);
        return view;
    }
}
