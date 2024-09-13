#ifndef FTV_HPP
#define FTV_HPP

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <Eigen/Geometry>
#include <tf2_eigen/tf2_eigen.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include "core/eye.hpp"
//#include "executor.hpp"
//#include "config/configuration.hpp"
#include "common/state/state.hpp"
#include "task2.hpp"

#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <geometric_shapes/shape_operations.h>



class FTVNode final : public rclcpp::Node {
public:
    FTVNode() : Node("ftv_ros"), custom_origin_(Eigen::Isometry3d::Identity()) {
        RCLCPP_INFO(this->get_logger(), "Initializing FTVNode");
        std::cout << "Current Path: " << std::filesystem::current_path() << std::endl;
        RCLCPP_INFO(this->get_logger(), "Current Path: %s", std::filesystem::current_path().c_str());
        initializeConfiguration();
        setCustomOrigin();
        init_timer_ = this->create_wall_timer(std::chrono::seconds(1), [this] {
            initializeMoveGroup();
            RCLCPP_INFO(this->get_logger(), "Initialized MoveGroupInterface");
        });
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("visualization_marker_array", 10);
        publishGroundPlaneMarker();
        RCLCPP_INFO(this->get_logger(), "FTVNode initialized successfully!");
    }

private:
    void initializeConfiguration() const {
        try {
            const auto package_share_directory = ament_index_cpp::get_package_share_directory("ftv");
            const auto config_file_path = package_share_directory + "/config/configuration.yaml";
            RCLCPP_INFO(this->get_logger(), "Loading configuration from: %s", config_file_path.c_str());
            // config::initialize(config_file_path);
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize configuration: %s", e.what());
        }
    }

    void setCustomOrigin() {
        custom_origin_ = Eigen::Isometry3d::Identity();
        custom_origin_.translation() = Eigen::Vector3d(0.25, 0.0, 0.0);
        RCLCPP_INFO(this->get_logger(), "Custom origin set to (%.2f, %.2f, %.2f)",
                    custom_origin_.translation().x(),
                    custom_origin_.translation().y(),
                    custom_origin_.translation().z());
    }

    void initializeMoveGroup() {
        RCLCPP_INFO(this->get_logger(), "Initializing MoveGroupInterface");
        init_timer_->cancel();

        try {
            move_group_interface_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
                shared_from_this(), "xarm7");

            move_group_interface_->setPlannerId("RRTConnect");
            move_group_interface_->setNumPlanningAttempts(20);
            move_group_interface_->setPlanningTime(1.0);
            move_group_interface_->setMaxVelocityScalingFactor(1.0);
            move_group_interface_->setMaxAccelerationScalingFactor(1.0);
            move_group_interface_->setGoalPositionTolerance(0.01);
            move_group_interface_->setGoalOrientationTolerance(0.01);

            /*
            move_group_interface_->setPlanningTime(2.0);
            move_group_interface_->setNumPlanningAttempts(10);
            move_group_interface_->setGoalPositionTolerance(0.01);
            move_group_interface_->setGoalOrientationTolerance(0.01);
            move_group_interface_->setMaxVelocityScalingFactor(0.8);
            move_group_interface_->setMaxAccelerationScalingFactor(0.8);
            */

            setupScene();

            core::Eye::setCallback(
                [this](const Eigen::Matrix4d &extrinsics, std::condition_variable &cv, bool &ready_flag) {
                    moveArm(extrinsics);
                    ready_flag = true;
                    cv.notify_one();
                });

            std::thread core_thread([this] { runCoreProgram(); });
            core_thread.detach();

            RCLCPP_INFO(this->get_logger(), "MoveGroupInterface initialized successfully");
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize MoveGroupInterface: %s", e.what());
        }
    }

    void runCoreProgram() const {
        try {
            RCLCPP_INFO(this->get_logger(), "Running core program");
//            Executor::execute();
            task2::run_level_3();
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Core program execution failed: %s", e.what());
        }
    }

    void setupScene() {
        if (!move_group_interface_) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface not initialized");
            return;
        }

        moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

        // Add a ground plane
        moveit_msgs::msg::CollisionObject ground_plane;
        ground_plane.header.frame_id = "link_base";
        ground_plane.id = "ground_plane";

        shape_msgs::msg::Plane plane;

        /*
            Equation of a plane: ax + by + cz + d = 0
            {1, 0, 0, 0}: vertical plane at x = 0 (the YZ plane)
            {0, 1, 0, 0}: vertical plane at y = 0 (the XZ plane)
            {0, 0, 1, -1}: horizontal plane at z = 1
            {1, 1, 1, -1}: plane that intersects the x, y, and z axes at 1
         */
        plane.coef = {0, 0, 1, 0}; // Ground plane

        ground_plane.planes.push_back(plane);
        ground_plane.plane_poses.push_back(geometry_msgs::msg::Pose());

        planning_scene_interface.applyCollisionObject(ground_plane);

        // Add bounding box
        addBoundingBox(planning_scene_interface);

        RCLCPP_INFO(this->get_logger(), "Added ground plane and bounding box to the scene");
    }

    void addBoundingBox(moveit::planning_interface::PlanningSceneInterface& planning_scene_interface) {
        std::vector<moveit_msgs::msg::CollisionObject> collision_objects;

        // Define the dimensions of the bounding box
        const double x_min = -0.5, x_max = 1.0;
        const double y_min = -0.75, y_max = 0.75;
        const double z_min = 0.0, z_max = 1.5;

        // Create 6 planes to form a box
        std::vector<std::array<double, 4>> plane_coeffs = {
            {1, 0, 0, -x_min},  // Left wall
            {-1, 0, 0, x_max},  // Right wall
            {0, 1, 0, -y_min},  // Front wall
            {0, -1, 0, y_max},  // Back wall
            {0, 0, 1, -z_min},  // Floor (already added as ground plane but still)
            {0, 0, -1, z_max}   // Ceiling
        };

        for (size_t i = 0; i < plane_coeffs.size(); ++i) {
            moveit_msgs::msg::CollisionObject collision_object;
            collision_object.header.frame_id = "link_base";
            collision_object.id = "bounding_plane_" + std::to_string(i);

            shape_msgs::msg::Plane plane;
            plane.coef = {plane_coeffs[i][0], plane_coeffs[i][1], plane_coeffs[i][2], plane_coeffs[i][3]};

            collision_object.planes.push_back(plane);
            collision_object.plane_poses.push_back(geometry_msgs::msg::Pose());

            collision_objects.push_back(collision_object);
        }

        planning_scene_interface.applyCollisionObjects(collision_objects);

        // Visualize the bounding box
        publishBoundingBoxMarkers(x_min, x_max, y_min, y_max, z_min, z_max);
    }

    void publishBoundingBoxMarkers(double x_min, double x_max, double y_min, double y_max, double z_min, double z_max) const {
        visualization_msgs::msg::MarkerArray marker_array;

        auto createBoxMarker = [&](int id, double x, double y, double z, double dx, double dy, double dz) {
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "link_base";
            marker.header.stamp = this->now();
            marker.ns = "bounding_box";
            marker.id = id;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.pose.position.x = x;
            marker.pose.position.y = y;
            marker.pose.position.z = z;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = dx;
            marker.scale.y = dy;
            marker.scale.z = dz;
            marker.color.r = 0.5;
            marker.color.g = 0.5;
            marker.color.b = 0.5;
            marker.color.a = 0.2;
            return marker;
        };

        // Create markers for the edges of the bounding box
        const double edge_thickness = 0.02;
        marker_array.markers.push_back(createBoxMarker(0, (x_min + x_max) / 2, y_min, z_min, x_max - x_min, edge_thickness, edge_thickness));
        marker_array.markers.push_back(createBoxMarker(1, (x_min + x_max) / 2, y_max, z_min, x_max - x_min, edge_thickness, edge_thickness));
        marker_array.markers.push_back(createBoxMarker(2, (x_min + x_max) / 2, y_min, z_max, x_max - x_min, edge_thickness, edge_thickness));
        marker_array.markers.push_back(createBoxMarker(3, (x_min + x_max) / 2, y_max, z_max, x_max - x_min, edge_thickness, edge_thickness));
        marker_array.markers.push_back(createBoxMarker(4, x_min, (y_min + y_max) / 2, z_min, edge_thickness, y_max - y_min, edge_thickness));
        marker_array.markers.push_back(createBoxMarker(5, x_max, (y_min + y_max) / 2, z_min, edge_thickness, y_max - y_min, edge_thickness));
        marker_array.markers.push_back(createBoxMarker(6, x_min, (y_min + y_max) / 2, z_max, edge_thickness, y_max - y_min, edge_thickness));
        marker_array.markers.push_back(createBoxMarker(7, x_max, (y_min + y_max) / 2, z_max, edge_thickness, y_max - y_min, edge_thickness));
        marker_array.markers.push_back(createBoxMarker(8, x_min, y_min, (z_min + z_max) / 2, edge_thickness, edge_thickness, z_max - z_min));
        marker_array.markers.push_back(createBoxMarker(9, x_max, y_min, (z_min + z_max) / 2, edge_thickness, edge_thickness, z_max - z_min));
        marker_array.markers.push_back(createBoxMarker(10, x_min, y_max, (z_min + z_max) / 2, edge_thickness, edge_thickness, z_max - z_min));
        marker_array.markers.push_back(createBoxMarker(11, x_max, y_max, (z_min + z_max) / 2, edge_thickness, edge_thickness, z_max - z_min));

        marker_pub_->publish(marker_array);
    }

    void moveArm(const Eigen::Matrix4d &extrinsics) const {
        RCLCPP_INFO(this->get_logger(), "Attempting to move the arm");

        if (!move_group_interface_) {
            RCLCPP_ERROR(this->get_logger(), "MoveGroupInterface not initialized");
            return;
        }

        try {
            Eigen::Vector3d target_position = extrinsics.block<3, 1>(0, 3);
            Eigen::Vector3d origin = custom_origin_.translation();
            Eigen::Vector3d line_direction = (target_position - origin).normalized();

            double max_reach = 1; // TODO: tweak based on arm lenght
            const int max_retries = 50;
            const double min_step = 0.01;
            double step_size = max_reach;
            geometry_msgs::msg::Pose best_reachable_pose = move_group_interface_->getCurrentPose().pose;
            bool improvement_found = false;

            for (int retry = 0; retry < max_retries; ++retry) {
                Eigen::Vector3d test_position = origin + line_direction * step_size;
                geometry_msgs::msg::Pose test_pose = calculatePoseWithOrientation(test_position, origin);

                move_group_interface_->setPoseTarget(test_pose);
                moveit::planning_interface::MoveGroupInterface::Plan my_plan;
                auto plan_result = move_group_interface_->plan(my_plan);

                if (plan_result == moveit::core::MoveItErrorCode::SUCCESS) {
                    RCLCPP_INFO(this->get_logger(), "Found reachable point at distance: %f. Executing...", step_size);
                    auto execution_result = move_group_interface_->execute(my_plan);
                    if (execution_result == moveit::core::MoveItErrorCode::SUCCESS) {
                        best_reachable_pose = test_pose;
                        improvement_found = true;

                        // try to reach the target position from the new reachable point
                        if (step_size > (target_position - origin).norm()) {
                            geometry_msgs::msg::Pose target_pose =
                                    calculatePoseWithOrientation(target_position, origin);
                            move_group_interface_->setPoseTarget(target_pose);
                            plan_result = move_group_interface_->plan(my_plan);
                            if (plan_result == moveit::core::MoveItErrorCode::SUCCESS) {
                                execution_result = move_group_interface_->execute(my_plan);
                                if (execution_result == moveit::core::MoveItErrorCode::SUCCESS) {
                                    RCLCPP_INFO(this->get_logger(), "Successfully reached the original target.");
                                    best_reachable_pose = target_pose;
                                    break;
                                }
                            }
                        } else {
                            RCLCPP_INFO(this->get_logger(), "Reached best possible point along the line.");
                            break;
                        }
                    }
                }

                // Reduce step size
                step_size *= 0.8;
                if (step_size < min_step) {
                    RCLCPP_WARN(this->get_logger(), "Reached minimum step size. Using best found pose.");
                    break;
                }
            }

            if (improvement_found) {
                RCLCPP_INFO(this->get_logger(), "Moving to best reachable pose.");
                move_group_interface_->setPoseTarget(best_reachable_pose);
                move_group_interface_->move();
            } else {
                RCLCPP_WARN(this->get_logger(),
                            "Failed to find any reachable point. Attempting small movement in target direction.");
                // Attempt a small movement in the direction of the target
                auto current_pose = move_group_interface_->getCurrentPose().pose;
                Eigen::Vector3d current_position(current_pose.position.x, current_pose.position.y,
                                                 current_pose.position.z);
                Eigen::Vector3d small_step = (target_position - current_position).normalized() * 0.05;
                Eigen::Vector3d new_position = current_position + small_step;
                geometry_msgs::msg::Pose new_pose = calculatePoseWithOrientation(new_position, origin);
                move_group_interface_->setPoseTarget(new_pose);
                move_group_interface_->move();
            }

            // Publish visualization markers
            auto final_pose = move_group_interface_->getCurrentPose().pose;
            Eigen::Vector3d final_position(final_pose.position.x, final_pose.position.y, final_pose.position.z);
            publishVisualizationMarkers(final_position);

            // Marker for the target position
            publishTargetMarker(target_position);
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in moveArm: %s", e.what());
        }
    }

    geometry_msgs::msg::Pose calculatePoseWithOrientation(const Eigen::Vector3d &position,
                                                          const Eigen::Vector3d &target) const {
        Eigen::Vector3d direction = (target - position).normalized();

        // point end effector's z-axis to point towards the target (origin with offset)
        Eigen::Vector3d z_axis = direction;

        // using world up vector because we want a vector not parallel to z_axis
        Eigen::Vector3d world_up(0, 0, 1);

        // Calculate y_axis - perpendicuklar to both z axis and world up
        Eigen::Vector3d y_axis = z_axis.cross(world_up).normalized();

        // Recalculate x_axis to ensure orthogonality
        Eigen::Vector3d x_axis = y_axis.cross(z_axis).normalized();

        // Create rotation matrix
        Eigen::Matrix3d rotation;
        rotation.col(0) = x_axis;
        rotation.col(1) = y_axis;
        rotation.col(2) = z_axis;

        Eigen::Quaterniond orientation(rotation);

        geometry_msgs::msg::Pose pose;
        pose.position.x = position.x();
        pose.position.y = position.y();
        pose.position.z = position.z();
        pose.orientation = tf2::toMsg(orientation);

        return pose;
    }

    void publishTargetMarker(const Eigen::Vector3d &target_position) const {
        visualization_msgs::msg::Marker target_marker;
        target_marker.header.frame_id = "link_base";
        target_marker.header.stamp = this->now();
        target_marker.ns = "target_position";
        target_marker.id = 2;
        target_marker.type = visualization_msgs::msg::Marker::SPHERE;
        target_marker.action = visualization_msgs::msg::Marker::ADD;
        target_marker.pose.position.x = target_position.x();
        target_marker.pose.position.y = target_position.y();
        target_marker.pose.position.z = target_position.z();
        target_marker.scale.x = 0.05;
        target_marker.scale.y = 0.05;
        target_marker.scale.z = 0.05;
        target_marker.color.r = 0.0;
        target_marker.color.g = 0.0;
        target_marker.color.b = 1.0;
        target_marker.color.a = 1.0;

        visualization_msgs::msg::MarkerArray marker_array;
        marker_array.markers.push_back(target_marker);
        marker_pub_->publish(marker_array);
    }

    void setOrientationToLookAtOrigin(geometry_msgs::msg::Pose &pose,
                                      const Eigen::Isometry3d &current_pose_robot) const {
        try {
            Eigen::Vector3d origin(0.25, 0.0, 0.0); // Offset origin - object is at x=0.25, y=0, z=0
            Eigen::Vector3d current_position = current_pose_robot.translation();
            Eigen::Vector3d forward = (origin - current_position).normalized();

            // We want the end effector's z-axis to point towards the origin
            Eigen::Vector3d up = Eigen::Vector3d::UnitZ(); // World up vector
            Eigen::Vector3d right = up.cross(forward).normalized();
            up = forward.cross(right).normalized();

            // Create rotation matrix
            Eigen::Matrix3d rotation;
            rotation.col(0) = right;
            rotation.col(1) = up;
            rotation.col(2) = forward;

            // Convert to quaternion
            Eigen::Quaterniond q(rotation);

            // Set the orientation in the pose
            pose.orientation.x = q.x();
            pose.orientation.y = q.y();
            pose.orientation.z = q.z();
            pose.orientation.w = q.w();

            RCLCPP_INFO(this->get_logger(), "Set orientation to look at origin: [%f, %f, %f, %f]",
                        q.w(), q.x(), q.y(), q.z());
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in setOrientationToLookAtOrigin: %s", e.what());
        }
    }

    Eigen::Isometry3d poseToEigen(const geometry_msgs::msg::Pose &pose) const {
        Eigen::Isometry3d eigen_pose = Eigen::Isometry3d::Identity();
        eigen_pose.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
        Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
        eigen_pose.linear() = q.toRotationMatrix();
        return eigen_pose;
    }

    void publishVisualizationMarkers(const Eigen::Vector3d &target_position) const {
        visualization_msgs::msg::MarkerArray marker_array;

        // Marker for the target point
        visualization_msgs::msg::Marker point_marker;
        point_marker.header.frame_id = "link_base";
        point_marker.header.stamp = this->now();
        point_marker.ns = "target_point";
        point_marker.id = 0;
        point_marker.type = visualization_msgs::msg::Marker::SPHERE;
        point_marker.action = visualization_msgs::msg::Marker::ADD;
        point_marker.pose.position.x = target_position.x();
        point_marker.pose.position.y = target_position.y();
        point_marker.pose.position.z = target_position.z();
        point_marker.scale.x = 0.05;
        point_marker.scale.y = 0.05;
        point_marker.scale.z = 0.05;
        point_marker.color.r = 1.0;
        point_marker.color.g = 0.0;
        point_marker.color.b = 0.0;
        point_marker.color.a = 1.0;

        // Marker for the arrow pointing to the origin
        visualization_msgs::msg::Marker arrow_marker;
        arrow_marker.header.frame_id = "link_base";
        arrow_marker.header.stamp = this->now();
        arrow_marker.ns = "target_to_origin_arrow";
        arrow_marker.id = 1;
        arrow_marker.type = visualization_msgs::msg::Marker::ARROW;
        arrow_marker.action = visualization_msgs::msg::Marker::ADD;
        arrow_marker.points.resize(2);
        arrow_marker.points[0].x = target_position.x();
        arrow_marker.points[0].y = target_position.y();
        arrow_marker.points[0].z = target_position.z();
        arrow_marker.points[1].x = custom_origin_.translation().x();
        arrow_marker.points[1].y = custom_origin_.translation().y();
        arrow_marker.points[1].z = custom_origin_.translation().z();
        arrow_marker.scale.x = 0.01; // shaft diameter
        arrow_marker.scale.y = 0.02; // head diameter
        arrow_marker.scale.z = 0.1; // head length
        arrow_marker.color.r = 0.0;
        arrow_marker.color.g = 1.0;
        arrow_marker.color.b = 0.0;
        arrow_marker.color.a = 1.0;

        marker_array.markers.push_back(point_marker);
        marker_array.markers.push_back(arrow_marker);

        marker_pub_->publish(marker_array);
    }

    void publishGroundPlaneMarker() const {
        visualization_msgs::msg::Marker ground_marker;
        ground_marker.header.frame_id = "link_base";
        ground_marker.header.stamp = this->now();
        ground_marker.ns = "ground_plane";
        ground_marker.id = 3;
        ground_marker.type = visualization_msgs::msg::Marker::CUBE;
        ground_marker.action = visualization_msgs::msg::Marker::ADD;
        ground_marker.pose.position.x = 0;
        ground_marker.pose.position.y = 0;
        ground_marker.pose.position.z = -0.005; // Slightly below z=0 to avoid z-fighting
        ground_marker.pose.orientation.w = 1.0;
        ground_marker.scale.x = 2.0;
        ground_marker.scale.y = 2.0;
        ground_marker.scale.z = 0.01;
        ground_marker.color.r = 0.8;
        ground_marker.color.g = 0.8;
        ground_marker.color.b = 0.8;
        ground_marker.color.a = 0.5;

        visualization_msgs::msg::MarkerArray marker_array;
        marker_array.markers.push_back(ground_marker);
        marker_pub_->publish(marker_array);
    }

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_interface_;
    rclcpp::TimerBase::SharedPtr init_timer_;
    Eigen::Isometry3d custom_origin_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

#endif //FTV_HPP
