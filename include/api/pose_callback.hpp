// File: api/pose_callback.hpp

#ifndef POSE_CALLBACK_HPP
#define POSE_CALLBACK_HPP

#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include "common/logging/logger.hpp"
#include "interface/callback.hpp"
#include "types/viewpoint.hpp"

class PoseCallback : public Callback {
public:
    void registerCallback(CallbackFunction callback) override {
        std::lock_guard lock(mutex_);
        callbacks_.push_back(std::move(callback));
        LOG_INFO("New pose callback registered. Total callbacks: {}", callbacks_.size());
    }

    void invoke(const ViewPoint<> &viewpoint) const override {
        std::lock_guard lock(mutex_);
        for (const auto &callback: callbacks_) {
            callback(viewpoint);
        }
        LOG_DEBUG("Pose callback invoked for viewpoint: {}", viewpoint.toString());
    }

private:
    mutable std::mutex mutex_;
    std::vector<CallbackFunction> callbacks_;
};

#endif // POSE_CALLBACK_HPP
