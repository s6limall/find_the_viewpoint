// File: robot_perception.hpp

#ifndef ROBOT_PERCEPTION_HPP
#define ROBOT_PERCEPTION_HPP

#include "core/vision/perception.hpp"
#include "core/camera.hpp"
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/pose.hpp>
#include <mutex>
#include <condition_variable>

class RobotPerception final : public core::Perception {
public:
    explicit RobotPerception(const rclcpp::Node::SharedPtr& node);
    ~RobotPerception() override = default;

    [[nodiscard]] cv::Mat render(const Eigen::Matrix4d &extrinsics, std::string_view save_path) override;
    [[nodiscard]] std::shared_ptr<core::Camera> getCamera() const noexcept override;

private:
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    cv::Mat latest_image_;
    std::mutex mutex_;
    std::condition_variable cv_;

    bool moveToPose(const geometry_msgs::msg::Pose& target_pose);
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    static geometry_msgs::msg::Pose convertEigenToPose(const Eigen::Matrix4d &extrinsics);
};

#endif // ROBOT_PERCEPTION_HPP
