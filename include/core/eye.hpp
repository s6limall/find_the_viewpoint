// File: core/eye.hpp

#ifndef EYE_HPP
#define EYE_HPP

#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include "common/logging/logger.hpp"
#include "config/configuration.hpp"
#include "core/vision/perception.hpp"
#include "core/vision/robot.hpp"
#include "core/vision/simulator.hpp"
#include "interface/pose_publisher.hpp"

namespace core {

    class Eye {
    public:
        // Rule of five
        Eye(const Eye &) = default;
        Eye &operator=(const Eye &) = default;
        Eye(Eye &&) noexcept = default;
        Eye &operator=(Eye &&) noexcept = default;
        ~Eye() = default;

        static void initialize() {
            std::string perception_type = config::get("perception.type", "simulator");
            if (perception_type == "simulator") {
                perception_ = std::make_unique<Simulator>();
            } else if (perception_type == "robot") {
                // perception_ = std::make_unique<Robot>();
            } else {
                LOG_ERROR("Invalid perception type in configuration: {}", perception_type);
                throw std::runtime_error("Invalid perception type in configuration");
            }
        }

        [[nodiscard]] static cv::Mat render(const Eigen::Matrix4d &extrinsics,
                                            const std::string_view image_save_path = "capture.jpg") {
            std::call_once(init_flag_, &Eye::initialize);

            cv::Mat result = perception_->render(extrinsics, image_save_path);

            Camera::Extrinsics extrinsics_matrix;
            extrinsics_matrix.setPose(extrinsics);

            // Publish the pose after rendering
            if (pose_publisher_) {
                const ViewPoint<> viewpoint(extrinsics_matrix.getTranslation());
                pose_publisher_->publishPose(viewpoint);
            }

            return result;
        }

        [[nodiscard]] static std::shared_ptr<Camera> getCamera() {
            std::call_once(init_flag_, &Eye::initialize);
            return perception_->getCamera();
        }

        static void setPosePublisher(std::shared_ptr<PosePublisher> publisher) {
            std::call_once(init_flag_, &Eye::initialize);
            pose_publisher_ = std::move(publisher);
        }

    private:
        inline static std::unique_ptr<Perception> perception_ = nullptr;
        inline static std::shared_ptr<PosePublisher> pose_publisher_ = nullptr;
        inline static std::once_flag init_flag_;
    };

} // namespace core

#endif // EYE_HPP
