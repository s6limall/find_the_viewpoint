// File: robot_perception.cpp

#include "robot_perception.hpp"
#include <tf2_eigen/tf2_eigen.hpp>

RobotPerception::RobotPerception(const rclcpp::Node::SharedPtr &node)
    : node_(node) {
    RCLCPP_INFO(node_->get_logger(), "Initializing RobotPerception");

    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "xarm7");

    // Initialize the camera
    camera_ = std::make_shared<core::Camera>();
    camera_->setIntrinsics(640, 480, 0.95, 0.75);
    camera_->setPosition(1.0, 0, 1.0);
    camera_->lookAt(Eigen::Vector3d(0, 0, 0));

    // Subscribe to the simulated camera topic
    image_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", 10,
        std::bind(&RobotPerception::imageCallback, this, std::placeholders::_1));

    RCLCPP_INFO(node_->get_logger(), "RobotPerception initialized successfully");
}

cv::Mat RobotPerception::render(const Eigen::Matrix4d &extrinsics, const std::string_view save_path) {
    geometry_msgs::msg::Pose target_pose = convertEigenToPose(extrinsics);

    if (moveToPose(target_pose)) {
        // Wait for a new image to be received
        std::unique_lock<std::mutex> lock(mutex_);
        if (cv_.wait_for(lock, std::chrono::seconds(5), [this] { return !latest_image_.empty(); })) {
            cv::Mat image = latest_image_.clone();

            if (!save_path.empty()) {
                cv::imwrite(std::string(save_path), image);
            }

            return image;
        } else {
            RCLCPP_ERROR(node_->get_logger(), "Timeout waiting for image after movement.");
        }
    }

    return cv::Mat();
}

std::shared_ptr<core::Camera> RobotPerception::getCamera() const noexcept {
    return camera_;
}

bool RobotPerception::moveToPose(const geometry_msgs::msg::Pose &target_pose) {
    move_group_->setPoseTarget(target_pose);
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    bool success = (move_group_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

    if (success) {
        RCLCPP_INFO(node_->get_logger(), "Successfully planned movement to the target pose.");
        moveit::core::MoveItErrorCode result = move_group_->execute(my_plan);
        return result == moveit::core::MoveItErrorCode::SUCCESS;
    } else {
        RCLCPP_ERROR(node_->get_logger(), "Failed to plan movement to the target pose.");
        return false;
    }
}

void RobotPerception::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_image_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
    cv_.notify_one();
}

geometry_msgs::msg::Pose RobotPerception::convertEigenToPose(const Eigen::Matrix4d &extrinsics) {
    Eigen::Isometry3d eigen_pose(extrinsics);
    geometry_msgs::msg::Pose pose;
    tf2::convert(eigen_pose, pose);
    return pose;
}
