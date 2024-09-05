// File: ftv.hpp

#ifndef FTV_HPP
#define FTV_HPP

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "task2.hpp"
#include "common/state/state.hpp"

class FTVNode final : public rclcpp::Node {
public:
    FTVNode() : Node("ftv_ros") {
        RCLCPP_INFO(this->get_logger(), "Initializing FTVNode");

        std::cout << "Current Path: " << std::filesystem::current_path() << std::endl;
        // Initialize configuration
        initializeConfiguration();

        // Set up a timer to initialize MoveGroupInterface after the node is fully set up
        init_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            [this] { initializeMoveGroup(); });
    }

    void initializeMoveGroup() {
        RCLCPP_INFO(this->get_logger(), "Initializing MoveGroupInterface");
        init_timer_->cancel();

        try {
            move_group_interface_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                shared_from_this(), "xarm7");

            // Set default planning parameters for speed
            move_group_interface_->setPlanningTime(2);
            move_group_interface_->setNumPlanningAttempts(15);
            move_group_interface_->setGoalPositionTolerance(0.5);
            move_group_interface_->setGoalOrientationTolerance(0.5);
            move_group_interface_->setMaxVelocityScalingFactor(1.0);
            move_group_interface_->setMaxAccelerationScalingFactor(1.0);


            // Set up the Eye callback
            core::Eye::setCallback(
                [this](const Eigen::Matrix4d &extrinsics, std::condition_variable &cv, bool &ready_flag) {
                    moveArm(extrinsics);
                    ready_flag = true;
                    cv.notify_one();
                });

            // Run the core program
            std::thread core_thread([this] { run(); });
            core_thread.detach();

            RCLCPP_INFO(this->get_logger(), "MoveGroupInterface initialized successfully");
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize MoveGroupInterface: %s", e.what());
        }
    }

private:
    void initializeConfiguration() const {
        // Get the package share directory
        const auto package_share_directory = ament_index_cpp::get_package_share_directory("ftv");

        // Set the configuration file path
        const auto config_file_path = package_share_directory + "/config/configuration.yaml";
        RCLCPP_INFO(this->get_logger(), "Loading configuration from: %s", config_file_path.c_str());

        // Load the configuration
        // config::initialize(config_file_path);

        // Set the paths for resources
        /*state::set("paths.mesh", package_share_directory + "/meshes");
        state::set("paths.target_images", package_share_directory + "/core/target_images");
        state::set("paths.3d_models", package_share_directory + "/models/3d_models");*/
    }

    void run() const {
        RCLCPP_INFO(this->get_logger(), "Running core program");
        // Executor::execute();
        task2::run_level_3();
    }

    void moveArm(const Eigen::Matrix4d &extrinsics) const {
        RCLCPP_INFO(this->get_logger(), "Attempting to move the arm");

        if (!move_group_interface_) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface not initialized");
            return;
        }

        // Convert Eigen::Matrix4d to geometry_msgs::msg::Pose
        geometry_msgs::msg::Pose target_pose;
        const Eigen::Affine3d affine(extrinsics);
        Eigen::Quaterniond q(affine.rotation());

        target_pose.position.x = extrinsics(0, 3);
        target_pose.position.y = extrinsics(1, 3);
        target_pose.position.z = extrinsics(2, 3);
        target_pose.orientation.x = q.x();
        target_pose.orientation.y = q.y();
        target_pose.orientation.z = q.z();
        target_pose.orientation.w = q.w();

        // Set planning parameters
        move_group_interface_->setPlanningTime(5.0); // 5 seconds planning time
        move_group_interface_->setNumPlanningAttempts(100);
        move_group_interface_->setGoalPositionTolerance(0.5); // 1cm
        move_group_interface_->setGoalOrientationTolerance(0.5); // ~5.7 degrees


        // Try to plan to the exact target
        move_group_interface_->setPoseTarget(target_pose);
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        bool success = (move_group_interface_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

        if (!success) {
            RCLCPP_WARN(this->get_logger(),
                        "Unable to plan to exact target. Attempting to find nearest reachable point.");

            // Get the current end effector pose
            const geometry_msgs::msg::PoseStamped current_pose = move_group_interface_->getCurrentPose();

            // Calculate direction vector from current pose to target pose
            Eigen::Vector3d direction(
                target_pose.position.x - current_pose.pose.position.x,
                target_pose.position.y - current_pose.pose.position.y,
                target_pose.position.z - current_pose.pose.position.z
            );

            // Normalize the direction vector
            direction.normalize();

            // Try different distances along the direction vector
            for (double distance = 1.0; distance > 0.1; distance -= 0.1) {
                geometry_msgs::msg::Pose test_pose = target_pose;
                test_pose.position.x = current_pose.pose.position.x + direction.x() * distance;
                test_pose.position.y = current_pose.pose.position.y + direction.y() * distance;
                test_pose.position.z = current_pose.pose.position.z + direction.z() * distance;

                move_group_interface_->setPoseTarget(test_pose);
                success = (move_group_interface_->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);

                if (success) {
                    RCLCPP_INFO(this->get_logger(), "Found reachable point at distance: %f", distance);
                    break;
                }
            }
        }

        if (success) {
            RCLCPP_INFO(this->get_logger(), "Planning successful. Executing...");
            move_group_interface_->execute(my_plan);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Unable to find a reachable point.");
        }
    }

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_interface_;
    rclcpp::TimerBase::SharedPtr init_timer_;
};

#endif //FTV_HPP
