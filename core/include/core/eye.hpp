// File: core/eye.hpp

#ifndef EYE_HPP
#define EYE_HPP

#include <Eigen/Core>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include "common/logging/logger.hpp"
#include "common/state/state.hpp"
// #include "config/configuration.hpp"
#include "core/vision/perception.hpp"
#include "core/vision/simulator.hpp"

namespace core {

    class Eye {
    public:
        using ExtrinsicsCallback =
                std::function<void(const Eigen::Matrix4d &extrinsics, std::condition_variable &cv, bool &ready_flag)>;

        // Delete constructor to enforce static usage
        Eye() = delete;

        // Rule of five (default implementation since it's a static class)
        Eye(const Eye &) = default;
        Eye &operator=(const Eye &) = default;
        Eye(Eye &&) noexcept = default;
        Eye &operator=(Eye &&) noexcept = default;
        ~Eye() = default;

        // Static method to initialize the perception system
        static void initialize() {
            // std::string perception_type = config::get("perception.type", "simulator");
            const std::string perception_type = state::get("perception.type", "simulator");
            if (perception_type == "simulator") {
                const auto mesh_path =
                        state::get("paths.mesh", "./3d_models/obj_000020.ply");
                perception_ = core::Simulator::create(mesh_path);
            } else if (perception_type == "robot") {
                LOG_WARN("Robot perception not set.");
            } else {
                LOG_ERROR("Invalid perception type in configuration: {}", perception_type);
                throw std::runtime_error("Invalid perception type in configuration");
            }
        }

        // Static method to render the image based on the extrinsics
        [[nodiscard]] static cv::Mat render(const Eigen::Matrix4d &extrinsics,
                                            const std::string_view image_save_path = "capture.jpg") {
            std::call_once(init_flag_, &Eye::initialize);

            if (callback_.has_value()) {
                callback_.value()(extrinsics, cv_, ready_flag_);
                std::unique_lock<std::mutex> cv_lock(mutex_);
                cv_.wait(cv_lock, [] { return ready_flag_; });
            }


            cv::Mat result = perception_->render(extrinsics, image_save_path);
            state::set("count", state::get("count", 0) + 1);

            Camera::Extrinsics extrinsics_matrix;
            extrinsics_matrix.setPose(extrinsics);

            return result;
        }

        // Static method to get the camera from the perception system
        [[nodiscard]] static std::shared_ptr<Camera> getCamera() {
            std::call_once(init_flag_, &Eye::initialize);
            return perception_->getCamera();
        }

        // Static method to set the perception system (simulator, robot)
        static void setPerception(std::shared_ptr<Perception> perception) {
            std::call_once(init_flag_, &Eye::initialize);
            perception_ = std::move(perception);
        }

        static std::shared_ptr<Perception> getPerception() {
            std::call_once(init_flag_, &Eye::initialize);
            return perception_;
        }

        static void setCallback(ExtrinsicsCallback callback) { callback_ = std::move(callback); }


    private:
        inline static std::shared_ptr<Perception> perception_ = nullptr;
        inline static std::once_flag init_flag_;
        inline static std::mutex mutex_;
        inline static std::condition_variable cv_;
        inline static bool ready_flag_ = false;
        inline static std::optional<ExtrinsicsCallback> callback_ = std::nullopt;
    };

} // namespace core

#endif // EYE_HPP
