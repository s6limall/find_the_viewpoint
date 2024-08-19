/*
// File: core/vision/robot.hpp

#ifndef ROBOT_HPP
#define ROBOT_HPP

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "common/logging/logger.hpp"
#include "core/vision/perception.hpp"

#include <cv_bridge/cv_bridge.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace core {
    class Robot final : public Perception, public rclcpp::Node {
    public:
        Robot();
        ~Robot() override = default;

        [[nodiscard]] cv::Mat render(const Eigen::Matrix4d &extrinsics, std::string_view save_path) override;
        [[nodiscard]] std::shared_ptr<Camera> getCamunera() override { return camera_; }

    private:
        void initialize();
        std::shared_ptr<Camera> camera_;

        // ROS2 and MoveIt components
        std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
        std::shared_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;

        cv::Mat latest_image_;
        std::mutex image_mutex_;

        void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
        bool moveRobotToPosition(const Eigen::Matrix4d &extrinsics);
    };
} // namespace core

#endif // ROBOT_HPP
*/
