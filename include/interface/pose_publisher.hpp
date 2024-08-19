// File: interface/pose_publisher.hpp

#ifndef POSE_PUBLISHER_HPP
#define POSE_PUBLISHER_HPP

#include <memory>

#include "common/logging/logger.hpp"
#include "interface/pose_callback.hpp"
#include "types/viewpoint.hpp"

class PosePublisher {
public:
    explicit PosePublisher(std::shared_ptr<PoseCallback> callback) : callback_(std::move(callback)) {
        if (!callback_) {
            throw std::invalid_argument("PoseCallback cannot be null");
        }
    }

    void publishPose(const ViewPoint<> &viewpoint) const {
        callback_->invoke(viewpoint);
        LOG_INFO("Published pose: {}", viewpoint.toString());
    }

private:
    std::shared_ptr<PoseCallback> callback_;
};


#endif // POSE_PUBLISHER_HPP
