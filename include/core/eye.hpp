// File: core/eye.hpp

#ifndef EYE_HPP
#define EYE_HPP

#include <Eigen/Core>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include "api/interface/publisher.hpp"
#include "common/logging/logger.hpp"
#include "common/state/state.hpp"
#include "config/configuration.hpp"
#include "core/vision/perception.hpp"
#include "core/vision/simulator.hpp"

namespace core {

    class Eye {
    public:
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
            std::string perception_type = config::get("perception.type", "simulator");
            if (perception_type == "simulator") {
                const auto mesh_path =
                        state::get("paths.mesh", config::get("paths.mesh", "./3d_models/obj_000020.ply"));
                perception_ = core::Simulator::create(mesh_path);
                // perception_ = Simulator::create();
            } else if (perception_type == "robot") {
                // perception_ = std::make_unique<Robot>();
                LOG_ERROR("Robot perception not yet implemented.");
                throw std::runtime_error("Robot perception not implemented");
            } else {
                LOG_ERROR("Invalid perception type in configuration: {}", perception_type);
                throw std::runtime_error("Invalid perception type in configuration");
            }
        }

        // Static method to render the image based on the extrinsics
        [[nodiscard]] static cv::Mat render(const Eigen::Matrix4d &extrinsics,
                                            const std::string_view image_save_path = "capture.jpg") {
            std::call_once(init_flag_, &Eye::initialize);

            cv::Mat result = perception_->render(extrinsics, image_save_path);

            Camera::Extrinsics extrinsics_matrix;
            extrinsics_matrix.setPose(extrinsics);

            // Publish the pose after rendering
            if (pose_publisher_) {
                const ViewPoint<> viewpoint(extrinsics_matrix.getTranslation());
                pose_publisher_->publish(viewpoint);
            }

            return result;
        }

        // Static method to get the camera from the perception system
        [[nodiscard]] static std::shared_ptr<Camera> getCamera() {
            std::call_once(init_flag_, &Eye::initialize);
            return perception_->getCamera();
        }

        // Static method to set the pose publisher
        static void setPosePublisher(std::shared_ptr<Publisher> publisher) {
            std::call_once(init_flag_, &Eye::initialize);
            pose_publisher_ = std::move(publisher);
        }

        // Static method to set the perception system (simulator, robot)
        static void setPerception(std::shared_ptr<Perception> perception) {
            std::call_once(init_flag_, &Eye::initialize);
            perception_ = std::move(perception);
        }

    private:
        inline static std::shared_ptr<Perception> perception_ = nullptr;
        inline static std::shared_ptr<Publisher> pose_publisher_ = nullptr;
        inline static std::once_flag init_flag_;
    };

} // namespace core

#endif // EYE_HPP
