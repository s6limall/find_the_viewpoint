// File: api/pose_publisher.hpp

#ifndef POSE_PUBLISHER_HPP
#define POSE_PUBLISHER_HPP

#include <memory>

#include "common/logging/logger.hpp"
#include "api/pose_callback.hpp"
#include "interface/publisher.hpp"
#include "types/viewpoint.hpp"

class PosePublisher : public Publisher {
public:
    explicit PosePublisher(std::shared_ptr<PoseCallback> callback) : callback_(std::move(callback)) {
        if (!callback_) {
            throw std::invalid_argument("PoseCallback cannot be null");
        }
    }

    void publish(const ViewPoint<> &viewpoint) const override {
        callback_->invoke(viewpoint);
        LOG_INFO("Published pose: {}", viewpoint.toString());
    }

private:
    std::shared_ptr<PoseCallback> callback_;
};


#endif // POSE_PUBLISHER_HPP
