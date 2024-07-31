#include <ma_scvp_real/ros_interface.h>

#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <eigen_conversions/eigen_msg.h>

RosInterface::RosInterface(ros::NodeHandle& nh):
    nh_(nh),
    target_frame_("base_link")
{
    // initialize tf listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>();
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

    octomap_pub_ = nh.advertise<octomap_msgs::Octomap>("octomap", 1, true);

    // initialize pointcloud publisher
    pc_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("pointcloud", 1, true);

    // initialize rgb publisher
    rgb_pub_ = nh_.advertise<sensor_msgs::Image>("rgb", 1, true);

    pose_array_pub_ = nh_.advertise<geometry_msgs::PoseArray>("pose_array", 1, true);

    path_pub_ = nh_.advertise<nav_msgs::Path>("path", 1, true);
    global_path_pub_ = nh_.advertise<nav_msgs::Path>("global_path", 1, true);
}

RosInterface::~RosInterface(){
    ;
}

void RosInterface::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& point_cloud_msg){
    ROS_INFO("Pointcloud received");
    // unsubscribe to stop receiving messages
    point_cloud_sub_.shutdown();

    // pre-process point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc = preProcessPointCloud(point_cloud_msg);
    
    *share_data->cloud_now = *pc;

    // set flag
    is_point_cloud_received_ = true;
}

void RosInterface::rgbCallback(const sensor_msgs::ImageConstPtr& rgb_msg){
    ROS_INFO("RGB image received");
    // unsubscribe to stop receiving messages
    rgb_sub_.shutdown();

    // convert to cv::Mat
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("RGB image conversion failed: %s", e.what());
        return;
    }

    // convert to cv::Mat
    cv::Mat rgb = cv_ptr->image;

    share_data->rgb_now = rgb.clone();

    // set flag
    is_rgb_image_received_ = true;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr RosInterface::preProcessPointCloud(const sensor_msgs::PointCloud2ConstPtr &point_cloud_msg){
    // transform pointcloud msg from camera link to target_frame
    sensor_msgs::PointCloud2 transformed_msg;
    transformed_msg.header.frame_id = target_frame_;

    if (tf_buffer_)
    {
        geometry_msgs::TransformStamped transform_stamped;
        try
        {
            transform_stamped = tf_buffer_->lookupTransform(target_frame_, point_cloud_msg->header.frame_id, ros::Time(0));

            tf2::doTransform(*point_cloud_msg, transformed_msg, transform_stamped);
        }
        catch (tf2::TransformException &ex)
        {
            ROS_ERROR("PCL transform failed: %s", ex.what());
            ROS_ERROR("Unable to transform pointcloud from %s to %s", point_cloud_msg->header.frame_id.c_str(), target_frame_.c_str());
            ROS_ERROR("Make sure that the transform between these two frames is published");
            
            return pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
        }
    } else {
        ROS_ERROR_THROTTLE(2.0, "tf_buffer_ is not initialized");
        return pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
    }

    // Convert pointcloud to pcl::PointCloud
    pcl::PCLPointCloud2Ptr pc2(new pcl::PCLPointCloud2);
    pcl_conversions::toPCL(transformed_msg, *pc2);
    pc2->header.frame_id = transformed_msg.header.frame_id;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(*pc2, *pc);

    ROS_INFO("Pointcloud received and transformed to %s", target_frame_.c_str());

    // Publish to rviz
    // sensor_msgs::PointCloud2 pc2_msg;
    // pcl::toROSMsg(*pc, pc2_msg);
    // pc2_msg.header.frame_id = target_frame_;
    // pc2_msg.header.stamp = ros::Time::now();
    // pc_pub_.publish(pc2_msg);

    // ROS_INFO("Pointcloud published to rviz");
    // ros::Duration(2.0).sleep();
    
    return pc;
}

void RosInterface::run()
{
    initMoveitClient();

    // wait for 2 seconds
    ros::Duration(1.0).sleep();

    // create 2 waypoints from the current pose
    geometry_msgs::PoseStamped current_pose = getCurrentPose();

    // initialize planner
    share_data = make_shared<Share_Data>("/home/user/pan/PRV/src/NBV_Simulation_MA-SCVP/DefaultConfiguration.yaml");

    // check for pose
    if (share_data->mode == -1){
        //record jiont pose
        geometry_msgs::PoseStamped current_pose1 = getCurrentPose();
        vector<double> joint_values = getJointValues();
        cout << "joint_values: ";
        for (auto& ptr : joint_values) {
            cout << ptr << ' ';
        }
        cout << endl;

        return;
    }
    
    // point start matrix back: -0.064299, 0.212141, 0.274420, 0.727306, 0.628116, -0.213824, 0.175429
    // point start matrix left: 0.257573, 0.492113, 0.244797, -0.104466, 0.959358, 0.208935, 0.158321
    // point start matrix front: -0.003004, 0.678657, 0.201165, 0.174885, 0.145978, -0.730413, 0.643897
    // point start matrix right: -0.318362, 0.443421, 0.167907, 0.986109, -0.009298, -0.164120, 0.023821

    // initialize planner to get object pos and size
    if (share_data->mode == 0) { //test object pos and size
        cout << "Test object pos and size" << endl;

        //move to three choosed start point
        auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->setBackgroundColor(255, 255, 255);
        viewer->addCoordinateSystem(0.1);
        viewer->initCameraParameters();

        //first point start matrix top: -0.073738, 0.457149, 0.431199, -0.002172, 0.684334, -0.011654, 0.729073
        double start_x = -0.073738;
        double start_y = 0.457149;
        double start_z = 0.431199;
        double start_qx = -0.002172;
        double start_qy = 0.684334;
        double start_qz = -0.011654;
        double start_qw = 0.729073;

        vector<vector<float>> now_waypoints1;
		vector<float> temp_waypoint1 = {float(start_x), float(start_y), float(start_z), float(start_qx), float(start_qy), float(start_qz), float(start_qw)};
        now_waypoints1.push_back(temp_waypoint1);
        std::vector<geometry_msgs::Pose> waypoints_msg1;
        generateWaypoints(now_waypoints1, waypoints_msg1);
        if (visitWaypoints(waypoints_msg1)){
            ROS_INFO("MoveitClient: Arm moved to waypoints");
        }
        else{
            ROS_ERROR("MoveitClient: Failed to move arm to waypoints");
        }
        geometry_msgs::PoseStamped current_pose1 = getCurrentPose();

        //show start point
        Eigen::Quaterniond start_q(start_qw, start_qx, start_qy, start_qz);
        Eigen::Matrix3d start_rotation = start_q.toRotationMatrix();
        Eigen::Matrix4d start_pose_world = Eigen::Matrix4d::Identity();
        start_pose_world.block<3,3>(0, 0) = start_rotation;
        start_pose_world(0, 3) = start_x;
        start_pose_world(1, 3) = start_y;
        start_pose_world(2, 3) = start_z;

        share_data->now_camera_pose_world = start_pose_world * share_data->camera_depth_to_rgb.inverse();

        Eigen::Vector4d X(0.05, 0, 0, 1);
        Eigen::Vector4d Y(0, 0.05, 0, 1);
        Eigen::Vector4d Z(0, 0, 0.05, 1);
        Eigen::Vector4d O(0, 0, 0, 1);
        X = share_data->now_camera_pose_world * X;
        Y = share_data->now_camera_pose_world * Y;
        Z = share_data->now_camera_pose_world * Z;
        O = share_data->now_camera_pose_world * O;
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(-1));
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(-1));
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(-1));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(-1));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(-1));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(-1));

        //get pointcloud and init ground truth
        getPointCloud();
        while(is_point_cloud_received_ == false){
            ros::Duration(0.2).sleep();
        }
        is_point_cloud_received_ = false;
        *share_data->cloud_scene += *share_data->cloud_now;
        std::cout << "point cloud scene first size: " << share_data->cloud_scene->size() << std::endl;

        // point start matrix back: -0.064299, 0.212141, 0.274420, 0.727306, 0.628116, -0.213824, 0.175429
        double start_x2 = -0.064299;
        double start_y2 = 0.212141;
        double start_z2 = 0.274420;
        double start_qx2 = 0.727306;
        double start_qy2 = 0.628116;
        double start_qz2 = -0.213824;
        double start_qw2 = 0.175429;

        vector<vector<float>> now_waypoints2;
        vector<float> temp_waypoint2 = {float(start_x2), float(start_y2), float(start_z2), float(start_qx2), float(start_qy2), float(start_qz2), float(start_qw2)};
        now_waypoints2.push_back(temp_waypoint2);
        std::vector<geometry_msgs::Pose> waypoints_msg2;
        generateWaypoints(now_waypoints2, waypoints_msg2);
        if (visitWaypoints(waypoints_msg2)){
            ROS_INFO("MoveitClient: Arm moved to waypoints");
        }
        else{
            ROS_ERROR("MoveitClient: Failed to move arm to waypoints");
        }
        geometry_msgs::PoseStamped current_pose2 = getCurrentPose();

        //show start point
        Eigen::Quaterniond start_q2(start_qw2, start_qx2, start_qy2, start_qz2);
        Eigen::Matrix3d start_rotation2 = start_q2.toRotationMatrix();
        Eigen::Matrix4d start_pose_world2 = Eigen::Matrix4d::Identity();
        start_pose_world2.block<3,3>(0, 0) = start_rotation2;
        start_pose_world2(0, 3) = start_x2;
        start_pose_world2(1, 3) = start_y2;
        start_pose_world2(2, 3) = start_z2;

        share_data->now_camera_pose_world = start_pose_world2 * share_data->camera_depth_to_rgb.inverse();

        Eigen::Vector4d X2(0.05, 0, 0, 1);
        Eigen::Vector4d Y2(0, 0.05, 0, 1);
        Eigen::Vector4d Z2(0, 0, 0.05, 1);
        Eigen::Vector4d O2(0, 0, 0, 1);
        X2 = share_data->now_camera_pose_world * X2;
        Y2 = share_data->now_camera_pose_world * Y2;
        Z2 = share_data->now_camera_pose_world * Z2;
        O2 = share_data->now_camera_pose_world * O2;
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O2(0), O2(1), O2(2)), pcl::PointXYZ(X2(0), X2(1), X2(2)), 255, 0, 0, "X" + to_string(-2));
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O2(0), O2(1), O2(2)), pcl::PointXYZ(Y2(0), Y2(1), Y2(2)), 0, 255, 0, "Y" + to_string(-2));
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O2(0), O2(1), O2(2)), pcl::PointXYZ(Z2(0), Z2(1), Z2(2)), 0, 0, 255, "Z" + to_string(-2));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(-2));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(-2));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(-2));

        //get pointcloud and init ground truth
        getPointCloud();
        while(is_point_cloud_received_ == false){
            ros::Duration(0.2).sleep();
        }
        is_point_cloud_received_ = false;

        *share_data->cloud_scene += *share_data->cloud_now;
        std::cout << "point cloud scene plus second size: " << share_data->cloud_scene->size() << std::endl;

        // point start matrix left: 0.257573, 0.492113, 0.244797, -0.104466, 0.959358, 0.208935, 0.158321
        double start_x3 = 0.257573;
        double start_y3 = 0.492113;
        double start_z3 = 0.244797;
        double start_qx3 = -0.104466;
        double start_qy3 = 0.959358;
        double start_qz3 = 0.208935;
        double start_qw3 = 0.158321;

        vector<vector<float>> now_waypoints3;
        vector<float> temp_waypoint3 = {float(start_x3), float(start_y3), float(start_z3), float(start_qx3), float(start_qy3), float(start_qz3), float(start_qw3)};
        now_waypoints3.push_back(temp_waypoint3);
        std::vector<geometry_msgs::Pose> waypoints_msg3;
        generateWaypoints(now_waypoints3, waypoints_msg3);
        if (visitWaypoints(waypoints_msg3)){
            ROS_INFO("MoveitClient: Arm moved to waypoints");
        }
        else{
            ROS_ERROR("MoveitClient: Failed to move arm to waypoints");
        }
        geometry_msgs::PoseStamped current_pose3 = getCurrentPose();

        //show start point
        Eigen::Quaterniond start_q3(start_qw3, start_qx3, start_qy3, start_qz3);
        Eigen::Matrix3d start_rotation3 = start_q3.toRotationMatrix();
        Eigen::Matrix4d start_pose_world3 = Eigen::Matrix4d::Identity();
        start_pose_world3.block<3,3>(0, 0) = start_rotation3;
        start_pose_world3(0, 3) = start_x3;
        start_pose_world3(1, 3) = start_y3;
        start_pose_world3(2, 3) = start_z3;

        share_data->now_camera_pose_world = start_pose_world3 * share_data->camera_depth_to_rgb.inverse();

        Eigen::Vector4d X3(0.05, 0, 0, 1);
        Eigen::Vector4d Y3(0, 0.05, 0, 1);
        Eigen::Vector4d Z3(0, 0, 0.05, 1);
        Eigen::Vector4d O3(0, 0, 0, 1);
        X3 = share_data->now_camera_pose_world * X3;
        Y3 = share_data->now_camera_pose_world * Y3;
        Z3 = share_data->now_camera_pose_world * Z3;
        O3 = share_data->now_camera_pose_world * O3;
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O3(0), O3(1), O3(2)), pcl::PointXYZ(X3(0), X3(1), X3(2)), 255, 0, 0, "X" + to_string(-3));
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O3(0), O3(1), O3(2)), pcl::PointXYZ(Y3(0), Y3(1), Y3(2)), 0, 255, 0, "Y" + to_string(-3));
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O3(0), O3(1), O3(2)), pcl::PointXYZ(Z3(0), Z3(1), Z3(2)), 0, 0, 255, "Z" + to_string(-3));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(-3));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(-3));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(-3));

        //get pointcloud and init ground truth
        getPointCloud();
        while(is_point_cloud_received_ == false){
            ros::Duration(0.2).sleep();
        }
        is_point_cloud_received_ = false;
        *share_data->cloud_scene += *share_data->cloud_now;
        std::cout << "point cloud scene plus third size: " << share_data->cloud_scene->size() << std::endl;

        // point start matrix front: -0.003004, 0.678657, 0.201165, 0.174885, 0.145978, -0.730413, 0.643897
        double start_x4 = -0.003004;
        double start_y4 = 0.678657;
        double start_z4 = 0.201165;
        double start_qx4 = 0.174885;
        double start_qy4 = 0.145978;
        double start_qz4 = -0.730413;
        double start_qw4 = 0.643897;

        vector<vector<float>> now_waypoints4;
        vector<float> temp_waypoint4 = {float(start_x4), float(start_y4), float(start_z4), float(start_qx4), float(start_qy4), float(start_qz4), float(start_qw4)};
        now_waypoints4.push_back(temp_waypoint4);
        std::vector<geometry_msgs::Pose> waypoints_msg4;
        generateWaypoints(now_waypoints4, waypoints_msg4);
        if (visitWaypoints(waypoints_msg4)){
            ROS_INFO("MoveitClient: Arm moved to waypoints");
        }
        else{
            ROS_ERROR("MoveitClient: Failed to move arm to waypoints");
        }
        geometry_msgs::PoseStamped current_pose4 = getCurrentPose();
        
        //show start point
        Eigen::Quaterniond start_q4(start_qw4, start_qx4, start_qy4, start_qz4);
        Eigen::Matrix3d start_rotation4 = start_q4.toRotationMatrix();
        Eigen::Matrix4d start_pose_world4 = Eigen::Matrix4d::Identity();
        start_pose_world4.block<3,3>(0, 0) = start_rotation4;
        start_pose_world4(0, 3) = start_x4;
        start_pose_world4(1, 3) = start_y4;
        start_pose_world4(2, 3) = start_z4;

        share_data->now_camera_pose_world = start_pose_world4 * share_data->camera_depth_to_rgb.inverse();

        Eigen::Vector4d X4(0.05, 0, 0, 1);
        Eigen::Vector4d Y4(0, 0.05, 0, 1);
        Eigen::Vector4d Z4(0, 0, 0.05, 1);
        Eigen::Vector4d O4(0, 0, 0, 1);
        X4 = share_data->now_camera_pose_world * X4;
        Y4 = share_data->now_camera_pose_world * Y4;
        Z4 = share_data->now_camera_pose_world * Z4;
        O4 = share_data->now_camera_pose_world * O4;
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O4(0), O4(1), O4(2)), pcl::PointXYZ(X4(0), X4(1), X4(2)), 255, 0, 0, "X" + to_string(-4));
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O4(0), O4(1), O4(2)), pcl::PointXYZ(Y4(0), Y4(1), Y4(2)), 0, 255, 0, "Y" + to_string(-4));
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O4(0), O4(1), O4(2)), pcl::PointXYZ(Z4(0), Z4(1), Z4(2)), 0, 0, 255, "Z" + to_string(-4));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(-4));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(-4));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(-4));

        //get pointcloud and init ground truth
        getPointCloud();
        while(is_point_cloud_received_ == false){
            ros::Duration(0.2).sleep();
        }
        is_point_cloud_received_ = false;
        *share_data->cloud_scene += *share_data->cloud_now;
        std::cout << "point cloud scene plus fourth size: " << share_data->cloud_scene->size() << std::endl;

        // point start matrix right: -0.318362, 0.443421, 0.167907, 0.986109, -0.009298, -0.164120, 0.023821
        double start_x5 = -0.318362;
        double start_y5 = 0.443421;
        double start_z5 = 0.167907;
        double start_qx5 = 0.986109;
        double start_qy5 = -0.009298;
        double start_qz5 = -0.164120;
        double start_qw5 = 0.023821;

        vector<vector<float>> now_waypoints5;
        vector<float> temp_waypoint5 = {float(start_x5), float(start_y5), float(start_z5), float(start_qx5), float(start_qy5), float(start_qz5), float(start_qw5)};
        now_waypoints5.push_back(temp_waypoint5);
        std::vector<geometry_msgs::Pose> waypoints_msg5;
        generateWaypoints(now_waypoints5, waypoints_msg5);
        if (visitWaypoints(waypoints_msg5)){
            ROS_INFO("MoveitClient: Arm moved to waypoints");
        }
        else{
            ROS_ERROR("MoveitClient: Failed to move arm to waypoints");
        }
        geometry_msgs::PoseStamped current_pose5 = getCurrentPose();

        //show start point
        Eigen::Quaterniond start_q5(start_qw5, start_qx5, start_qy5, start_qz5);
        Eigen::Matrix3d start_rotation5 = start_q5.toRotationMatrix();
        Eigen::Matrix4d start_pose_world5 = Eigen::Matrix4d::Identity();
        start_pose_world5.block<3,3>(0, 0) = start_rotation5;
        start_pose_world5(0, 3) = start_x5;
        start_pose_world5(1, 3) = start_y5;
        start_pose_world5(2, 3) = start_z5;

        share_data->now_camera_pose_world = start_pose_world5 * share_data->camera_depth_to_rgb.inverse();

        Eigen::Vector4d X5(0.05, 0, 0, 1);
        Eigen::Vector4d Y5(0, 0.05, 0, 1);
        Eigen::Vector4d Z5(0, 0, 0.05, 1);
        Eigen::Vector4d O5(0, 0, 0, 1);
        X5 = share_data->now_camera_pose_world * X5;
        Y5 = share_data->now_camera_pose_world * Y5;
        Z5 = share_data->now_camera_pose_world * Z5;
        O5 = share_data->now_camera_pose_world * O5;
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O5(0), O5(1), O5(2)), pcl::PointXYZ(X5(0), X5(1), X5(2)), 255, 0, 0, "X" + to_string(-5));
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O5(0), O5(1), O5(2)), pcl::PointXYZ(Y5(0), Y5(1), Y5(2)), 0, 255, 0, "Y" + to_string(-5));
        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O5(0), O5(1), O5(2)), pcl::PointXYZ(Z5(0), Z5(1), Z5(2)), 0, 0, 255, "Z" + to_string(-5));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(-5));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(-5));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(-5));

        //get pointcloud and init ground truth
        getPointCloud();
        while(is_point_cloud_received_ == false){
            ros::Duration(0.2).sleep();
        }
        is_point_cloud_received_ = false;
        *share_data->cloud_scene += *share_data->cloud_now;
        std::cout << "point cloud scene plus fifth size: " << share_data->cloud_scene->size() << std::endl;


        // pass through table
        pcl::PassThrough<pcl::PointXYZRGB> pass;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr no_table(new pcl::PointCloud<pcl::PointXYZRGB>);
        *no_table = *share_data->cloud_scene;
		pass.setInputCloud(no_table);          
        pass.setFilterFieldName("x");         
        pass.setFilterLimits(-0.25, 0.15);       
        pass.filter(*no_table);
        pass.setInputCloud(no_table);         
        pass.setFilterFieldName("y");         
        pass.setFilterLimits(0.25, 0.60);          
        pass.filter(*no_table);
        pass.setInputCloud(no_table);             
        pass.setFilterFieldName("z");   
		pass.setFilterLimits(share_data->height_of_ground, share_data->height_to_filter_arm); 
        pass.filter(*no_table);
        //对场景点云计算中心和大小
        vector<Eigen::Vector3d> points;
        for (auto& ptr : no_table->points) {
            Eigen::Vector3d point(ptr.x, ptr.y, ptr.z);
            points.push_back(point);
        }
        Eigen::Vector3d object_center_world = Eigen::Vector3d(0, 0, 0);
		for (auto& ptr : points) {
			object_center_world(0) += ptr(0);
			object_center_world(1) += ptr(1);
			object_center_world(2) += ptr(2);
		}
		object_center_world(0) /= points.size();
		object_center_world(1) /= points.size();
		object_center_world(2) /= points.size();
        cout << "object_center_world:" << object_center_world << endl;
		double predicted_size = 0.0;
		for (auto& ptr : points) {
			predicted_size = max(predicted_size, (object_center_world - ptr).norm());
		}
        cout << "predicted_size:" << predicted_size << endl;

        //save object center and size to file
        share_data->access_directory(share_data->pre_path);
        ofstream out_size(share_data->pre_path + share_data->name_of_pcd + "_centre_size.txt");
        out_size << "object_center_world:" << endl;
        out_size << object_center_world(0) << endl;
        out_size << object_center_world(1) << endl;
        out_size << object_center_world(2) << endl;
        out_size << "predicted_size:" << endl;
        out_size << predicted_size << endl;
        out_size.close();

        //save table and no table point cloud
        pcl::io::savePCDFileASCII(share_data->pre_path + share_data->name_of_pcd + ".pcd", *share_data->cloud_scene);
        pcl::io::savePCDFileASCII(share_data->pre_path + share_data->name_of_pcd + "_no_table.pcd", *no_table);

        //show object center and size
        viewer->addPointCloud<pcl::PointXYZRGB>(no_table, "no_table");

        while(viewer->wasStopped() == false){
            viewer->spinOnce(100);
            ros::Duration(0.1).sleep();
        }

        return;
    }

    // initialize planner to do ground truth planning
    if(share_data->mode == 1){ //gt_mode
        cout<<"Ground truth mode"<<endl;

        //遍历所有可能的视点空间的视点
        vector<int> requried_view_spaces;
        for(int i=3;i<=50;i+=2){
            requried_view_spaces.push_back(i);
        }
        // requried_view_spaces.push_back(5);
        requried_view_spaces.push_back(100);
        requried_view_spaces.push_back(144);
        
        int prv_vaule = 35;
        requried_view_spaces.push_back(prv_vaule);
        // 45, 49 min_z_table: 0.035 otherwise 0.075

        for(auto vs_it = requried_view_spaces.begin(); vs_it!=requried_view_spaces.end();vs_it++){
            //回到起点
            //first point start matrix -0.073738, 0.457149, 0.431199, -0.002172, 0.684334, -0.011654, 0.729073
            double start_x = -0.073738;
            double start_y = 0.457149;
            double start_z = 0.431199;
            double start_qx = -0.002172;
            double start_qy = 0.684334;
            double start_qz = -0.011654;
            double start_qw = 0.729073;

            if (moveArmToHome()){
                ROS_INFO("MoveitClient: Arm moved to home position");
            }
            else{
                ROS_ERROR("MoveitClient: Failed to move arm to home position");
            }
            ros::Duration(2.0).sleep();

            geometry_msgs::PoseStamped current_pose1 = getCurrentPose();

            //show start point
            Eigen::Quaterniond start_q(start_qw, start_qx, start_qy, start_qz);
            Eigen::Matrix3d start_rotation = start_q.toRotationMatrix();
            Eigen::Matrix4d start_pose_world = Eigen::Matrix4d::Identity();
            start_pose_world.block<3,3>(0, 0) = start_rotation;
            start_pose_world(0, 3) = start_x;
            start_pose_world(1, 3) = start_y;
            start_pose_world(2, 3) = start_z;

            share_data->now_camera_pose_world = start_pose_world * share_data->camera_depth_to_rgb.inverse();

            auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("3D Viewer"));
            viewer->setBackgroundColor(255, 255, 255);
            viewer->addCoordinateSystem(0.1);
            viewer->initCameraParameters();

            Eigen::Vector4d X(0.05, 0, 0, 1);
            Eigen::Vector4d Y(0, 0.05, 0, 1);
            Eigen::Vector4d Z(0, 0, 0.05, 1);
            Eigen::Vector4d O(0, 0, 0, 1);
            X = share_data->now_camera_pose_world * X;
            Y = share_data->now_camera_pose_world * Y;
            Z = share_data->now_camera_pose_world * Z;
            O = share_data->now_camera_pose_world * O;
            viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(-1));
            viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(-1));
            viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(-1));
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(-1));
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(-1));
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(-1));

            //获取view_space_size
            int view_space_size = *vs_it;
            cout<<"testing view_space_size: "<<view_space_size<<endl;
            
            //获取top view
            shared_ptr<Share_Data> share_data_initviews = make_shared<Share_Data>("/home/user/pan/PRV/src/NBV_Simulation_MA-SCVP/DefaultConfiguration.yaml", "", view_space_size);
            shared_ptr<View_Space> view_space_initviews = make_shared<View_Space>(share_data_initviews);
            vector<View> init_views = view_space_initviews->views;
            //get top view id
            int first_init_view_id = -1;
            for (int i = 0; i < share_data_initviews->pt_sphere.size(); i++) {
                Eigen::Vector3d test_pos = Eigen::Vector3d(share_data_initviews->pt_sphere[i][0] / share_data_initviews->pt_norm, share_data_initviews->pt_sphere[i][1] / share_data_initviews->pt_norm, share_data_initviews->pt_sphere[i][2] / share_data_initviews->pt_norm);
                if (fabs(test_pos(0)) < 1e-6 && fabs(test_pos(1)) < 1e-6 && fabs(test_pos(2) - 1.0) < 1e-6) {
                    first_init_view_id = i;
                }
            }
            if (first_init_view_id == -1) {
                cout << "can not find top view id" << endl;
            }

            viewer->setCameraPosition(0.8, 0.8, 0.8, share_data->object_center_world(0), share_data->object_center_world(1), share_data->object_center_world(2), 0, 0, 1);

            int num_of_movable_views = 0;
            vector<vector<double>> view_space_pose;
            vector<vector<double>> view_space_xyzq;
            vector<cv::Mat> view_space_rgb;

            // move to first view
            Eigen::Vector3d now_view_xyz(Eigen::Vector3d(start_pose_world(0,3), start_pose_world(1,3), start_pose_world(2,3)));
            Eigen::Vector3d next_view_xyz(view_space_initviews->views[first_init_view_id].init_pos(0), view_space_initviews->views[first_init_view_id].init_pos(1), view_space_initviews->views[first_init_view_id].init_pos(2));
            vector<Eigen::Vector3d> points;
            int num_of_path = get_trajectory_xyz(points, now_view_xyz, next_view_xyz, share_data->object_center_world, share_data->predicted_size, share_data->move_dis_pre_point, 0.0);
            if (num_of_path == -1) {
                cout << "no path. throw" << endl;
                return;
            }
            if (num_of_path == -2) cout << "Line" << endl;
            if (num_of_path > 0)  cout << "Obstcale" << endl;
            cout << "num_of_path:" << points.size() << endl;

            viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(now_view_xyz(0), now_view_xyz(1), now_view_xyz(2)), pcl::PointXYZ(points[0](0), points[0](1), points[0](2)), 0, 128, 128, "trajectory" + to_string(-1));
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory"  + to_string(-1));
            for (int k = 0; k < points.size() - 1; k++) {
                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(points[k](0), points[k](1), points[k](2)), pcl::PointXYZ(points[k + 1](0), points[k + 1](1), points[k + 1](2)), 0, 128, 128, "trajectory" + to_string(k));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory" + to_string(k));
            }

            // get waypoints
            vector<vector<float>> now_waypoints;
            Eigen::Matrix4d now_camera_pose_world = share_data->now_camera_pose_world;
            for(int j=0;j<points.size();j++){
                View temp_view(points[j]);
                temp_view.get_next_camera_pos(now_camera_pose_world,  share_data->object_center_world);
                Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

                cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
                if(j==points.size()-1){
                    Eigen::Vector4d X(0.05, 0, 0, 1);
                    Eigen::Vector4d Y(0, 0.05, 0, 1);
                    Eigen::Vector4d Z(0, 0, 0.05, 1);
                    Eigen::Vector4d O(0, 0, 0, 1);
                    X = now_camera_pose_world * temp_view.pose.inverse() * X;
                    Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                    Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                    O = now_camera_pose_world * temp_view.pose.inverse() * O;
                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(j));
                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(j));
                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(j));
                }

                Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                Eigen::Quaterniond temp_q(temp_rotation);
                vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
                now_waypoints.push_back(temp_waypoint);

                now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
            }

            viewer->spinOnce(100);

            std::vector<geometry_msgs::Pose> waypoints_msg;
            generateWaypoints(now_waypoints, waypoints_msg);
            if (visitWaypoints(waypoints_msg)){
                num_of_movable_views++;
                vector<double> current_joint = getJointValues();
                view_space_pose.push_back(current_joint);
                ROS_INFO("MoveitClient: Arm moved to waypoints");
            }
            else{
                ROS_ERROR("MoveitClient: Failed to move arm to waypoints");
            }
            geometry_msgs::PoseStamped current_pose = getCurrentPose();
            
            view_space_xyzq.push_back({current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z, current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z, current_pose.pose.orientation.w});
            share_data->now_camera_pose_world = now_camera_pose_world;

            ros::Duration(1.0).sleep();

            //获取RGB图像并保存
            getRGBImage();
            while(is_rgb_image_received_ == false){
                ros::Duration(0.2).sleep();
            }
            is_rgb_image_received_ = false;
            cv::Mat rgb_image = share_data->rgb_now;
            view_space_rgb.push_back(rgb_image);

            //计算全局路径
            vector<int> init_view_ids;
            for(int i=0;i<view_space_initviews->views.size();i++){
                init_view_ids.push_back(i);
            }
            Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data_initviews, init_views, init_view_ids, first_init_view_id);
            double init_dis = gloabl_path_planner->solve();
            init_view_ids = gloabl_path_planner->get_path_id_set();
            //无需反转
            //reverse(init_view_ids.begin(), init_view_ids.end());
            cout << "init_dis: " << init_dis << endl;
            delete gloabl_path_planner;

            // move to next view space
            for (int i = 0; i < init_view_ids.size() - 1; i++) {
                cout<< "view_space_pose.size():" << view_space_pose.size()<<endl;
                // otherwise move robot
                Eigen::Vector3d now_view_xyz(view_space_initviews->views[init_view_ids[i]].init_pos(0), view_space_initviews->views[init_view_ids[i]].init_pos(1), view_space_initviews->views[init_view_ids[i]].init_pos(2));
                Eigen::Vector3d next_view_xyz(view_space_initviews->views[init_view_ids[i+1]].init_pos(0), view_space_initviews->views[init_view_ids[i+1]].init_pos(1), view_space_initviews->views[init_view_ids[i+1]].init_pos(2));
                vector<Eigen::Vector3d> points;
                int num_of_path = get_trajectory_xyz(points, now_view_xyz, next_view_xyz, share_data->object_center_world, share_data->predicted_size, share_data->move_dis_pre_point, 0.0);
                if (num_of_path == -1) {
                    cout << "no path. throw" << endl;
                    return;
                }
                if (num_of_path == -2) cout << "Line" << endl;
                if (num_of_path > 0)  cout << "Obstcale" << endl;
                cout << "num_of_path:" << points.size() << endl;

                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(now_view_xyz(0), now_view_xyz(1), now_view_xyz(2)), pcl::PointXYZ(points[0](0), points[0](1), points[0](2)), 0, 128, 128, "trajectory" + to_string(i) + to_string(i + 1) + to_string(-1));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory" + to_string(i) + to_string(i + 1) + to_string(-1));
                for (int k = 0; k < points.size() - 1; k++) {
                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(points[k](0), points[k](1), points[k](2)), pcl::PointXYZ(points[k + 1](0), points[k + 1](1), points[k + 1](2)), 0, 128, 128, "trajectory" + to_string(i) + to_string(i + 1) + to_string(k));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory" + to_string(i) + to_string(i + 1) + to_string(k));
                }

                // get waypoints
                vector<vector<float>> now_waypoints;
                Eigen::Matrix4d now_camera_pose_world = share_data->now_camera_pose_world;
                for(int j=0;j<points.size();j++){
                    View temp_view(points[j]);
                    temp_view.get_next_camera_pos(now_camera_pose_world,  share_data->object_center_world);
                    Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

                    cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
                    if(j==points.size()-1){
                        Eigen::Vector4d X(0.05, 0, 0, 1);
                        Eigen::Vector4d Y(0, 0.05, 0, 1);
                        Eigen::Vector4d Z(0, 0, 0.05, 1);
                        Eigen::Vector4d O(0, 0, 0, 1);
                        X = now_camera_pose_world * temp_view.pose.inverse() * X;
                        Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                        Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                        O = now_camera_pose_world * temp_view.pose.inverse() * O;
                        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i)+  to_string(j));
                        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i) + to_string(j));
                        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i) + to_string(j));
                        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(i) + to_string(j));
                        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(i) + to_string(j));
                        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(i) + to_string(j));
                    }

                    Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                    Eigen::Quaterniond temp_q(temp_rotation);
                    vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
                    now_waypoints.push_back(temp_waypoint);

                    now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                }
                
                viewer->spinOnce(100);

                // try orginal path movement
                std::vector<geometry_msgs::Pose> waypoints_msg;
                generateWaypoints(now_waypoints, waypoints_msg);
                if (visitWaypoints(waypoints_msg)){
                    num_of_movable_views++;
                    vector<double> current_joint = getJointValues();
                    view_space_pose.push_back(current_joint);
                    ROS_INFO("MoveitClient: Arm moved to original waypoints 1");
                }
                else{
                    // try z table height path movement
                    ROS_ERROR("MoveitClient: Failed to move arm to original waypoints 1");
                    
                    // use the outside one to later update share_data->now_camera_pose_world
                    now_camera_pose_world = share_data->now_camera_pose_world;

                    vector<vector<float>> now_waypoints;
                    for(int j=0;j<points.size();j++){
                        View temp_view(points[j]);
                        Eigen::Vector3d now_object_center_world = share_data->object_center_world;
                        now_object_center_world(2) = share_data->min_z_table;
                        temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                        Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

                        cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
                        if(j==points.size()-1){
                            Eigen::Vector4d X(0.05, 0, 0, 1);
                            Eigen::Vector4d Y(0, 0.05, 0, 1);
                            Eigen::Vector4d Z(0, 0, 0.05, 1);
                            Eigen::Vector4d O(0, 0, 0, 1);
                            X = now_camera_pose_world * temp_view.pose.inverse() * X;
                            Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                            Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                            O = now_camera_pose_world * temp_view.pose.inverse() * O;

                            viewer->removeCorrespondences("X" + to_string(i) + to_string(j));
                            viewer->removeCorrespondences("Y" + to_string(i) + to_string(j));
                            viewer->removeCorrespondences("Z" + to_string(i) + to_string(j));

                            viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i)+  to_string(j));
                            viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i) + to_string(j));
                            viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i) + to_string(j));
                            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(i) + to_string(j));
                            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(i) + to_string(j));
                            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(i) + to_string(j));
                        }

                        Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                        Eigen::Quaterniond temp_q(temp_rotation);
                        vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
                        now_waypoints.push_back(temp_waypoint);

                        now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                    }

                    std::vector<geometry_msgs::Pose> waypoints_msg;
                    generateWaypoints(now_waypoints, waypoints_msg);
                    if (visitWaypoints(waypoints_msg)){
                        num_of_movable_views++;
                        vector<double> current_joint = getJointValues();
                        view_space_pose.push_back(current_joint);
                        ROS_INFO("MoveitClient: Arm moved to look at table height waypoints 2");
                    }
                    else{
                        ROS_ERROR("MoveitClient: Failed to move arm to look at table height waypoints 2");

                        // use the outside one to later update share_data->now_camera_pose_world
                        now_camera_pose_world = share_data->now_camera_pose_world;

                        //original point movement
                        View temp_view(view_space_initviews->views[init_view_ids[i+1]]);
                        Eigen::Vector3d now_object_center_world = share_data->object_center_world;
                        //now_object_center_world(2) = share_data->min_z_table;
                        temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                        Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 
                        Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                        Eigen::Quaterniond temp_q(temp_rotation);
                        vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };

                        // geometry_msgs::Pose target_orginal_pose;
                        // target_orginal_pose.position.x = temp_waypoint[0];
                        // target_orginal_pose.position.y = temp_waypoint[1];
                        // target_orginal_pose.position.z = temp_waypoint[2];
                        // target_orginal_pose.orientation.x = temp_waypoint[3];
                        // target_orginal_pose.orientation.y = temp_waypoint[4];
                        // target_orginal_pose.orientation.z = temp_waypoint[5];
                        // target_orginal_pose.orientation.w = temp_waypoint[6];
                        // cout << "target_orginal_pose:" << target_orginal_pose << endl;

                        // cout << "view_space_initviews->views[init_view_ids[i+1]]:" << view_space_initviews->views[init_view_ids[i+1]].init_pos << endl;
                        // cout << "temp_camera_pose_world:" << temp_camera_pose_world << endl;

                        vector<vector<float>> now_waypoints;
                        now_waypoints.push_back(temp_waypoint);
                        std::vector<geometry_msgs::Pose> waypoints_msg;
                        generateWaypoints(now_waypoints, waypoints_msg);

                        if (visitWaypoints(waypoints_msg)){
                            num_of_movable_views++;
                            vector<double> current_joint = getJointValues();
                            view_space_pose.push_back(current_joint);
                            ROS_INFO("MoveitClient: Arm moved to target original pose 3");
                            now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                        }
                        else{
                            ROS_ERROR("MoveitClient: Failed to move arm to original pose 3");

                            // use the outside one to later update share_data->now_camera_pose_world
                            now_camera_pose_world = share_data->now_camera_pose_world;

                            //table height point movement
                            View temp_view(view_space_initviews->views[init_view_ids[i+1]]);
                            Eigen::Vector3d now_object_center_world = share_data->object_center_world;
                            now_object_center_world(2) = share_data->min_z_table;
                            temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                            Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 
                            Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                            Eigen::Quaterniond temp_q(temp_rotation);
                            vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };

                            // geometry_msgs::Pose target_table_height_pose;
                            // target_table_height_pose.position.x = temp_waypoint[0];
                            // target_table_height_pose.position.y = temp_waypoint[1];
                            // target_table_height_pose.position.z = temp_waypoint[2];
                            // target_table_height_pose.orientation.x = temp_waypoint[3];
                            // target_table_height_pose.orientation.y = temp_waypoint[4];
                            // target_table_height_pose.orientation.z = temp_waypoint[5];
                            // target_table_height_pose.orientation.w = temp_waypoint[6];

                            vector<vector<float>> now_waypoints;
                            now_waypoints.push_back(temp_waypoint);
                            std::vector<geometry_msgs::Pose> waypoints_msg;
                            generateWaypoints(now_waypoints, waypoints_msg);

                            if (visitWaypoints(waypoints_msg)){
                                num_of_movable_views++;
                                vector<double> current_joint = getJointValues();
                                view_space_pose.push_back(current_joint);
                                ROS_INFO("MoveitClient: Arm moved to target look at table height pose 4");
                                now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                            }
                            else{
                                ROS_ERROR("MoveitClient: Failed to move arm to look at table height pose 4");
                                //go to home
                                if (moveArmToHome()){
                                    ROS_INFO("MoveitClient: Arm moved to home position");
                                }
                                else{
                                    ROS_ERROR("MoveitClient: Failed to move arm to home position");
                                }
                                //change now camera pose world to home
                                // -0.073738, 0.457149, 0.431199, -0.002172, 0.684334, -0.011654, 0.729073
                                double start_x = -0.073738;
                                double start_y = 0.457149;
                                double start_z = 0.431199;
                                double start_qx = -0.002172;
                                double start_qy = 0.684334;
                                double start_qz = -0.011654;
                                double start_qw = 0.729073;

                                Eigen::Quaterniond start_q(start_qw, start_qx, start_qy, start_qz);
                                Eigen::Matrix3d start_rotation = start_q.toRotationMatrix();
                                Eigen::Matrix4d start_pose_world = Eigen::Matrix4d::Identity();
                                start_pose_world.block<3,3>(0, 0) = start_rotation;
                                start_pose_world(0, 3) = start_x;
                                start_pose_world(1, 3) = start_y;
                                start_pose_world(2, 3) = start_z;

                                //update share_data->now_camera_pose_world
                                share_data->now_camera_pose_world = start_pose_world * share_data->camera_depth_to_rgb.inverse();

                                //use the outside one to later update share_data->now_camera_pose_world
                                now_camera_pose_world = share_data->now_camera_pose_world;
                                
                                Eigen::Vector3d now_view_xyz(now_camera_pose_world(0,3), now_camera_pose_world(1,3), now_camera_pose_world(2,3));
                                Eigen::Vector3d next_view_xyz(view_space_initviews->views[init_view_ids[i+1]].init_pos(0), view_space_initviews->views[init_view_ids[i+1]].init_pos(1), view_space_initviews->views[init_view_ids[i+1]].init_pos(2));
                                vector<Eigen::Vector3d> points;
                                int num_of_path = get_trajectory_xyz(points, now_view_xyz, next_view_xyz, share_data->object_center_world, share_data->predicted_size, share_data->move_dis_pre_point, 0.0);
                                if (num_of_path == -1) {
                                    cout << "no path. throw" << endl;
                                    return;
                                }
                                if (num_of_path == -2) cout << "Line" << endl;
                                if (num_of_path > 0)  cout << "Obstcale" << endl;
                                cout << "num_of_path:" << points.size() << endl;

                                // try two path again
                                vector<vector<float>> now_waypoints;
                                for(int j=0;j<points.size();j++){
                                    View temp_view(points[j]);
                                    temp_view.get_next_camera_pos(now_camera_pose_world,  share_data->object_center_world);
                                    Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

                                    cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
                                    if(j==points.size()-1){
                                        Eigen::Vector4d X(0.05, 0, 0, 1);
                                        Eigen::Vector4d Y(0, 0.05, 0, 1);
                                        Eigen::Vector4d Z(0, 0, 0.05, 1);
                                        Eigen::Vector4d O(0, 0, 0, 1);
                                        X = now_camera_pose_world * temp_view.pose.inverse() * X;
                                        Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                                        Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                                        O = now_camera_pose_world * temp_view.pose.inverse() * O;
                                        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i)+  to_string(j));
                                        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i) + to_string(j));
                                        viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i) + to_string(j));
                                        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(i) + to_string(j));
                                        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(i) + to_string(j));
                                        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(i) + to_string(j));
                                    }

                                    Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                                    Eigen::Quaterniond temp_q(temp_rotation);
                                    vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
                                    now_waypoints.push_back(temp_waypoint);

                                    now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                                }

                                // try orginal path movement from home
                                std::vector<geometry_msgs::Pose> waypoints_msg;
                                generateWaypoints(now_waypoints, waypoints_msg);
                                if (visitWaypoints(waypoints_msg)){
                                    num_of_movable_views++;
                                    vector<double> current_joint = getJointValues();
                                    view_space_pose.push_back(current_joint);
                                    ROS_INFO("MoveitClient: Arm moved to original waypoints from home 5");
                                }
                                else{
                                    // try z table height path movement from home
                                    ROS_ERROR("MoveitClient: Failed to move arm to original waypoints from home 5");
                                    
                                    // use the outside one to later update share_data->now_camera_pose_world
                                    now_camera_pose_world = share_data->now_camera_pose_world;

                                    vector<vector<float>> now_waypoints;
                                    for(int j=0;j<points.size();j++){
                                        View temp_view(points[j]);
                                        Eigen::Vector3d now_object_center_world = share_data->object_center_world;
                                        now_object_center_world(2) = share_data->min_z_table;
                                        temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                                        Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

                                        cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
                                        if(j==points.size()-1){
                                            Eigen::Vector4d X(0.05, 0, 0, 1);
                                            Eigen::Vector4d Y(0, 0.05, 0, 1);
                                            Eigen::Vector4d Z(0, 0, 0.05, 1);
                                            Eigen::Vector4d O(0, 0, 0, 1);
                                            X = now_camera_pose_world * temp_view.pose.inverse() * X;
                                            Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                                            Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                                            O = now_camera_pose_world * temp_view.pose.inverse() * O;

                                            viewer->removeCorrespondences("X" + to_string(i) + to_string(j));
                                            viewer->removeCorrespondences("Y" + to_string(i) + to_string(j));
                                            viewer->removeCorrespondences("Z" + to_string(i) + to_string(j));

                                            viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(i)+  to_string(j));
                                            viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(i) + to_string(j));
                                            viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(i) + to_string(j));
                                            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(i) + to_string(j));
                                            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(i) + to_string(j));
                                            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(i) + to_string(j));
                                        }

                                        Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                                        Eigen::Quaterniond temp_q(temp_rotation);
                                        vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
                                        now_waypoints.push_back(temp_waypoint);

                                        now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                                    }

                                    std::vector<geometry_msgs::Pose> waypoints_msg;
                                    generateWaypoints(now_waypoints, waypoints_msg);
                                    if (visitWaypoints(waypoints_msg)){
                                        num_of_movable_views++;
                                        vector<double> current_joint = getJointValues();
                                        view_space_pose.push_back(current_joint);
                                        ROS_INFO("MoveitClient: Arm moved to look at table height waypoints from home 6");
                                    }
                                    else{
                                        ROS_ERROR("MoveitClient: Failed to move arm to look at table height waypoints from home 6");

                                        now_camera_pose_world = share_data->now_camera_pose_world;

                                        //original point movement
                                        View temp_view(view_space_initviews->views[init_view_ids[i+1]]);
                                        Eigen::Vector3d now_object_center_world = share_data->object_center_world;
                                        //now_object_center_world(2) = share_data->min_z_table;
                                        temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                                        Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 
                                        Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                                        Eigen::Quaterniond temp_q(temp_rotation);
                                        vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };

                                        // geometry_msgs::Pose target_orginal_pose;
                                        // target_orginal_pose.position.x = temp_waypoint[0];
                                        // target_orginal_pose.position.y = temp_waypoint[1];
                                        // target_orginal_pose.position.z = temp_waypoint[2];
                                        // target_orginal_pose.orientation.x = temp_waypoint[3];
                                        // target_orginal_pose.orientation.y = temp_waypoint[4];
                                        // target_orginal_pose.orientation.z = temp_waypoint[5];
                                        // target_orginal_pose.orientation.w = temp_waypoint[6];

                                        vector<vector<float>> now_waypoints;
                                        now_waypoints.push_back(temp_waypoint);
                                        std::vector<geometry_msgs::Pose> waypoints_msg;
                                        generateWaypoints(now_waypoints, waypoints_msg);

                                        if (visitWaypoints(waypoints_msg)){
                                            num_of_movable_views++;
                                            vector<double> current_joint = getJointValues();
                                            view_space_pose.push_back(current_joint);
                                            ROS_INFO("MoveitClient: Arm moved to target original pose 7");
                                            now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                                        }
                                        else{
                                            ROS_ERROR("MoveitClient: Failed to move arm to original pose 7");

                                            now_camera_pose_world = share_data->now_camera_pose_world;

                                            //table height point movement
                                            View temp_view(view_space_initviews->views[init_view_ids[i+1]]);
                                            Eigen::Vector3d now_object_center_world = share_data->object_center_world;
                                            now_object_center_world(2) = share_data->min_z_table;
                                            temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                                            Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 
                                            Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                                            Eigen::Quaterniond temp_q(temp_rotation);
                                            vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };

                                            // geometry_msgs::Pose target_table_height_pose;
                                            // target_table_height_pose.position.x = temp_waypoint[0];
                                            // target_table_height_pose.position.y = temp_waypoint[1];
                                            // target_table_height_pose.position.z = temp_waypoint[2];
                                            // target_table_height_pose.orientation.x = temp_waypoint[3];
                                            // target_table_height_pose.orientation.y = temp_waypoint[4];
                                            // target_table_height_pose.orientation.z = temp_waypoint[5];
                                            // target_table_height_pose.orientation.w = temp_waypoint[6];

                                            vector<vector<float>> now_waypoints;
                                            now_waypoints.push_back(temp_waypoint);
                                            std::vector<geometry_msgs::Pose> waypoints_msg;
                                            generateWaypoints(now_waypoints, waypoints_msg);

                                            if (visitWaypoints(waypoints_msg)){
                                                num_of_movable_views++;
                                                vector<double> current_joint = getJointValues();
                                                view_space_pose.push_back(current_joint);
                                                ROS_INFO("MoveitClient: Arm moved to target look at table height pose 8");
                                                now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
                                            }
                                            else{
                                                ROS_ERROR("MoveitClient: Failed to move arm to look at table height pose 8");

                                                //go to home
                                                if (moveArmToHome()){
                                                    ROS_INFO("MoveitClient: Arm moved to home position");
                                                }
                                                else{
                                                    ROS_ERROR("MoveitClient: Failed to move arm to home position");
                                                }
                                                //change now camera pose world to home
                                                // -0.073738, 0.457149, 0.431199, -0.002172, 0.684334, -0.011654, 0.729073
                                                double start_x = -0.073738;
                                                double start_y = 0.457149;
                                                double start_z = 0.431199;
                                                double start_qx = -0.002172;
                                                double start_qy = 0.684334;
                                                double start_qz = -0.011654;
                                                double start_qw = 0.729073;

                                                Eigen::Quaterniond start_q(start_qw, start_qx, start_qy, start_qz);
                                                Eigen::Matrix3d start_rotation = start_q.toRotationMatrix();
                                                Eigen::Matrix4d start_pose_world = Eigen::Matrix4d::Identity();
                                                start_pose_world.block<3,3>(0, 0) = start_rotation;
                                                start_pose_world(0, 3) = start_x;
                                                start_pose_world(1, 3) = start_y;
                                                start_pose_world(2, 3) = start_z;

                                                now_camera_pose_world = start_pose_world * share_data->camera_depth_to_rgb.inverse();

                                                return;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                geometry_msgs::PoseStamped current_pose = getCurrentPose();

                //更新share_data->now_camera_pose_world
                double now_x = current_pose.pose.position.x;
                double now_y = current_pose.pose.position.y;
                double now_z = current_pose.pose.position.z;
                double now_qx = current_pose.pose.orientation.x;
                double now_qy = current_pose.pose.orientation.y;
                double now_qz = current_pose.pose.orientation.z;
                double now_qw = current_pose.pose.orientation.w;

                Eigen::Quaterniond now_q(now_qw, now_qx, now_qy, now_qz);
                Eigen::Matrix3d now_rotation = now_q.toRotationMatrix();
                Eigen::Matrix4d now_current_camera_pose_world = Eigen::Matrix4d::Identity();
                now_current_camera_pose_world.block<3,3>(0, 0) = now_rotation;
                now_current_camera_pose_world(0, 3) = now_x;
                now_current_camera_pose_world(1, 3) = now_y;
                now_current_camera_pose_world(2, 3) = now_z;

                share_data->now_camera_pose_world = now_current_camera_pose_world * share_data->camera_depth_to_rgb.inverse();

                ROS_INFO("moved views: %d", num_of_movable_views);

                //如果不是home的话, 按精度0.01对比
                if(fabs(current_pose.pose.position.x - (-0.073738)) > 0.01 || fabs(current_pose.pose.position.y - (0.457149)) > 0.01 || fabs(current_pose.pose.position.z - (0.431199)) > 0.01 || fabs(current_pose.pose.orientation.x - (-0.002172)) > 0.01 || fabs(current_pose.pose.orientation.y - (0.684334)) > 0.01 || fabs(current_pose.pose.orientation.z - (-0.011654)) > 0.01 || fabs(current_pose.pose.orientation.w - (0.729073)) > 0.01){
                    view_space_xyzq.push_back({current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z, current_pose.pose.orientation.x, current_pose.pose.orientation.y, current_pose.pose.orientation.z, current_pose.pose.orientation.w});
                    //获取RGB图像并保存
                    ros::Duration(1.0).sleep();
                    getRGBImage();
                    while(is_rgb_image_received_ == false){
                        ros::Duration(0.2).sleep();
                    }
                    is_rgb_image_received_ = false;
                    cv::Mat rgb_image = share_data->rgb_now;
                    view_space_rgb.push_back(rgb_image);
                }

                //如果末端轴很危险，就回到home
                bool dangerous_check = false;
                if (view_space_pose[view_space_pose.size()-1][3] < -4.71238898038469 || view_space_pose[view_space_pose.size()-1][3] > 4.71238898038469){
                    dangerous_check = true;
                }
                if (view_space_pose[view_space_pose.size()-1][4] < -4.71238898038469 || view_space_pose[view_space_pose.size()-1][4] > 4.71238898038469){
                    dangerous_check = true;
                }
                if(view_space_pose[view_space_pose.size()-1][5] < -4.71238898038469 || view_space_pose[view_space_pose.size()-1][5] > 4.71238898038469){
                    dangerous_check = true;
                }
                if(dangerous_check){
                    ROS_ERROR("MoveitClient: Arm moved to dangerous position, move to home position");
                    if (moveArmToHome()){
                        ROS_INFO("MoveitClient: Arm moved to home position");
                    }
                    else{
                        ROS_ERROR("MoveitClient: Failed to move arm to home position");
                    }

                    double start_x = -0.073738;
                    double start_y = 0.457149;
                    double start_z = 0.431199;
                    double start_qx = -0.002172;
                    double start_qy = 0.684334;
                    double start_qz = -0.011654;
                    double start_qw = 0.729073;

                    Eigen::Quaterniond start_q(start_qw, start_qx, start_qy, start_qz);
                    Eigen::Matrix3d start_rotation = start_q.toRotationMatrix();
                    Eigen::Matrix4d start_pose_world = Eigen::Matrix4d::Identity();
                    start_pose_world.block<3,3>(0, 0) = start_rotation;
                    start_pose_world(0, 3) = start_x;
                    start_pose_world(1, 3) = start_y;
                    start_pose_world(2, 3) = start_z;

                    share_data->now_camera_pose_world = start_pose_world * share_data->camera_depth_to_rgb.inverse();
                }

            }

            cout<<"num_of_movable_views: "<<num_of_movable_views<<endl;

            //关闭viewer
            if(num_of_movable_views != share_data_initviews->num_of_views){
                cout<<"num_of_movable_views != share_data->share_data_initviews"<<endl;
                while (!viewer->wasStopped()){
                    viewer->spinOnce(100);
                    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
                }
            }

            //保存到文件
            share_data->access_directory(share_data->viewspace_path + share_data->name_of_pcd + "/");

            //保存json与视点空间
            Json::Value root;
            root["camera_angle_x"] = 2.0 * atan(0.5 * share_data->color_intrinsics.width / share_data->color_intrinsics.fx);
            root["camera_angle_y"] = 2.0 * atan(0.5 * share_data->color_intrinsics.height / share_data->color_intrinsics.fy);
            root["fl_x"] = share_data->color_intrinsics.fx;
            root["fl_y"] = share_data->color_intrinsics.fy;
            root["k1"] = share_data->color_intrinsics.coeffs[0];
            root["k2"] = share_data->color_intrinsics.coeffs[1];
            root["k3"] = share_data->color_intrinsics.coeffs[2];
            root["p1"] = share_data->color_intrinsics.coeffs[3];
            root["p2"] = share_data->color_intrinsics.coeffs[4];
            root["cx"] = share_data->color_intrinsics.ppx;
            root["cy"] = share_data->color_intrinsics.ppy;
            root["w"] = share_data->color_intrinsics.width;
            root["h"] = share_data->color_intrinsics.height;
            root["aabb_scale"] = share_data->ray_casting_aabb_scale;
            root["scale"] = 1;
            root["offset"][0] = 0.5 + share_data->object_center_world(0) / share_data->view_space_radius;
            root["offset"][1] = 0.5;
            root["offset"][2] = 0.5;
            root["near_distance"] = (share_data->view_space_radius- share_data->predicted_size) / share_data->view_space_radius;

            ofstream fout_viewspace_pose(share_data->viewspace_path + share_data->name_of_pcd + "/" + to_string(share_data_initviews->num_of_views) + "_vs_pose.txt");
            ofstream fout_viewspace_xyzq(share_data->viewspace_path + share_data->name_of_pcd + "/" + to_string(share_data_initviews->num_of_views) + "_vs_xyzq.txt");
            
            for(int i = 0; i < view_space_pose.size(); i++){
                fout_viewspace_pose << init_view_ids[i] <<" " <<view_space_pose[i][0]<<" "<<view_space_pose[i][1]<<" "<<view_space_pose[i][2]<<" "<<view_space_pose[i][3]<<" "<<view_space_pose[i][4]<<" "<<view_space_pose[i][5]<<endl;
                fout_viewspace_xyzq << init_view_ids[i] <<" " <<view_space_xyzq[i][0] <<" "<< view_space_xyzq[i][1] <<" "<< view_space_xyzq[i][2] <<" "<< view_space_xyzq[i][3] <<" "<< view_space_xyzq[i][4] <<" "<< view_space_xyzq[i][5] <<" "<< view_space_xyzq[i][6] <<endl;

                double start_x = view_space_xyzq[i][0];
                double start_y = view_space_xyzq[i][1];
                double start_z = view_space_xyzq[i][2];
                double start_qx = view_space_xyzq[i][3];
                double start_qy = view_space_xyzq[i][4];
                double start_qz = view_space_xyzq[i][5];
                double start_qw = view_space_xyzq[i][6];

                //show start point
                Eigen::Quaterniond start_q(start_qw, start_qx, start_qy, start_qz);
                Eigen::Matrix3d start_rotation = start_q.toRotationMatrix();
                Eigen::Matrix4d start_pose_world = Eigen::Matrix4d::Identity();
                start_pose_world.block<3,3>(0, 0) = start_rotation;
                start_pose_world(0, 3) = start_x;
                start_pose_world(1, 3) = start_y;
                start_pose_world(2, 3) = start_z;

                Eigen::Matrix4d now_view_pose_world = start_pose_world * share_data->camera_depth_to_rgb.inverse();

                //get json
                Json::Value view_image;
                view_image["file_path"] = "../../Coverage_images/" + share_data->name_of_pcd + "/" + to_string(share_data_initviews->num_of_views) + "/rgbRembg_" + to_string(init_view_ids[i]) + ".png";
                Json::Value transform_matrix;
                for (int k = 0; k < 4; k++) {
                    Json::Value row;
                    for (int l = 0; l < 4; l++) {
                        Eigen::Matrix4d view_pose_world = now_view_pose_world;
                        //把视点空间的坐标系转换到json的坐标系，即移动到中心，然后缩放到1.0
                        view_pose_world(0, 3) = view_pose_world(0, 3) - share_data->object_center_world(0);
                        view_pose_world(1, 3) = view_pose_world(1, 3) - share_data->object_center_world(1);
                        view_pose_world(2, 3) = view_pose_world(2, 3) - share_data->object_center_world(2) - share_data->up_shift;
                        view_pose_world(0, 3) = view_pose_world(0, 3) / share_data->view_space_radius;
                        view_pose_world(1, 3) = view_pose_world(1, 3) / share_data->view_space_radius;
                        view_pose_world(2, 3) = view_pose_world(2, 3) / share_data->view_space_radius;
                        //x,y,z->y,z,x
                        Eigen::Matrix4d pose;
                        pose << 0, 0, 1, 0,
                            1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 0, 1;
                        //x,y,z->x,-y,-z
                        Eigen::Matrix4d pose_1;
                        pose_1 << 1, 0, 0, 0,
                            0, -1, 0, 0,
                            0, 0, -1, 0,
                            0, 0, 0, 1;
                        view_pose_world = pose * view_pose_world * pose_1;
                        row.append(view_pose_world(k, l));
                    }
                    transform_matrix.append(row);
			    }
                view_image["transform_matrix"] = transform_matrix;
                root["frames"].append(view_image);

            }
            fout_viewspace_pose.close();
            fout_viewspace_xyzq.close();

            Json::StyledWriter writer_json;
            ofstream fout_json(share_data->viewspace_path + share_data->name_of_pcd + "/" + to_string(share_data_initviews->num_of_views) + ".json");
            fout_json << writer_json.write(root);
            fout_json.close();

            //保存视点空间的RGB图像
            share_data->access_directory(share_data->gt_path + "/" + to_string(share_data_initviews->num_of_views) + "/");
            for(int i = 0; i < view_space_rgb.size(); i++){
                cv::imwrite(share_data->gt_path + "/" + to_string(share_data_initviews->num_of_views) + "/rgb_" + to_string(init_view_ids[i]) + ".png", view_space_rgb[i]);
            }

            //数据区清空
            share_data_initviews.reset();
            view_space_initviews.reset();

            while (!viewer->wasStopped()){
                viewer->spinOnce(100);
                boost::this_thread::sleep(boost::posix_time::microseconds(100000));
            }
            viewer->close();
        }

        cout<<"GT mode finish"<<endl;

        return;
    }

    // go to the home position
    cout<<"reconstruction mode"<<endl;

    //回到起点
    if (moveArmToHome()){
        ROS_INFO("MoveitClient: Arm moved to home position");
    }
    else{
        ROS_ERROR("MoveitClient: Failed to move arm to home position");
    }
    ros::Duration(1.0).sleep();

    double start_x = -0.073738;
    double start_y = 0.457149;
    double start_z = 0.431199;
    double start_qx = -0.002172;
    double start_qy = 0.684334;
    double start_qz = -0.011654;
    double start_qw = 0.729073;

    Eigen::Quaterniond start_q(start_qw, start_qx, start_qy, start_qz);
    Eigen::Matrix3d start_rotation = start_q.toRotationMatrix();
    Eigen::Matrix4d start_pose_world = Eigen::Matrix4d::Identity();
    start_pose_world.block<3,3>(0, 0) = start_rotation;
    start_pose_world(0, 3) = start_x;
    start_pose_world(1, 3) = start_y;
    start_pose_world(2, 3) = start_z;

    share_data->now_camera_pose_world = start_pose_world * share_data->camera_depth_to_rgb.inverse();

    vector<double> current_joints = getJointValues();

    double initial_time_stamp = ros::Time::now().toSec();

    int num_of_movable_views = 0;

    //set up the viewer and first view
    auto viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("Scene and Path"));
    viewer->setBackgroundColor(255, 255, 255);
    viewer->addCoordinateSystem(0.1);
    viewer->initCameraParameters();
    pcl::visualization::Camera cam;
    viewer->getCameraParameters(cam);
    cam.window_size[0] = 1920;
    cam.window_size[1] = 1080;
    viewer->setCameraParameters(cam);
    viewer->setCameraPosition(0.8, 0.8, 0.8, share_data->object_center_world(0), share_data->object_center_world(1), share_data->object_center_world(2), 0, 0, 1);

    //read init view space and go to init views
	int num_of_max_initviews = 5;
	vector<View> init_views;
	shared_ptr<Share_Data> share_data_initviews = make_shared<Share_Data>("/home/user/pan/PRV/src/NBV_Simulation_MA-SCVP/DefaultConfiguration.yaml", "", num_of_max_initviews);
	shared_ptr<View_Space> view_space_initviews = make_shared<View_Space>(share_data_initviews);
	init_views = view_space_initviews->views;
    //read finded view poses from file 
    ifstream fin_initviewspace_pose(share_data->viewspace_path + share_data->name_of_pcd + "/" + to_string(share_data_initviews->num_of_views) + "_vs_pose.txt");
    ifstream fin_initviewspace_xyzq(share_data->viewspace_path + share_data->name_of_pcd + "/" + to_string(share_data_initviews->num_of_views) + "_vs_xyzq.txt");
    vector<vector<double>> init_view_space_pose;
    vector<vector<double>> init_view_space_xyzq;
    init_view_space_pose.resize(share_data_initviews->num_of_views);
    init_view_space_xyzq.resize(share_data_initviews->num_of_views);
    for(int i = 0; i < share_data_initviews->num_of_views; i++){
        int view_id;
        fin_initviewspace_pose >> view_id;
        init_view_space_pose[view_id].resize(6);
        fin_initviewspace_pose>>init_view_space_pose[view_id][0]>>init_view_space_pose[view_id][1]>>init_view_space_pose[view_id][2]>>init_view_space_pose[view_id][3]>>init_view_space_pose[view_id][4]>>init_view_space_pose[view_id][5];
        fin_initviewspace_xyzq >> view_id;
        init_view_space_xyzq[view_id].resize(7);
        fin_initviewspace_xyzq>>init_view_space_xyzq[view_id][0]>>init_view_space_xyzq[view_id][1]>>init_view_space_xyzq[view_id][2]>>init_view_space_xyzq[view_id][3]>>init_view_space_xyzq[view_id][4]>>init_view_space_xyzq[view_id][5]>>init_view_space_xyzq[view_id][6];
        //转换为Eigen::Matrix4d
        double start_x = init_view_space_xyzq[view_id][0];
        double start_y = init_view_space_xyzq[view_id][1];
        double start_z = init_view_space_xyzq[view_id][2];
        double start_qx = init_view_space_xyzq[view_id][3];
        double start_qy = init_view_space_xyzq[view_id][4];
        double start_qz = init_view_space_xyzq[view_id][5];
        double start_qw = init_view_space_xyzq[view_id][6];
        Eigen::Quaterniond start_q(start_qw, start_qx, start_qy, start_qz);
        Eigen::Matrix3d start_rotation = start_q.toRotationMatrix();
        Eigen::Matrix4d start_pose_world = Eigen::Matrix4d::Identity();
        start_pose_world.block<3,3>(0, 0) = start_rotation;
        start_pose_world(0, 3) = start_x;
        start_pose_world(1, 3) = start_y;
        start_pose_world(2, 3) = start_z;
        Eigen::Matrix4d view_pose_world = start_pose_world * share_data->camera_depth_to_rgb.inverse();
        init_views[view_id].pose_fixed = view_pose_world;
    }
	//get top view id
	int first_init_view_id = -1;
	for (int i = 0; i < share_data_initviews->pt_sphere.size(); i++) {
		Eigen::Vector3d test_pos = Eigen::Vector3d(share_data_initviews->pt_sphere[i][0] / share_data_initviews->pt_norm, share_data_initviews->pt_sphere[i][1] / share_data_initviews->pt_norm, share_data_initviews->pt_sphere[i][2] / share_data_initviews->pt_norm);
		if (fabs(test_pos(0)) < 1e-6 && fabs(test_pos(1)) < 1e-6 && fabs(test_pos(2) - 1.0) < 1e-6) {
			first_init_view_id = i;
		}
	}
	if (first_init_view_id == -1) {
		cout << "can not find top view id" << endl;
	}
	vector<int> init_view_ids = share_data_initviews->init_view_ids;
	double init_dis = 0.0;
	if (init_view_ids.size() > 1) {
		Global_Path_Planner* gloabl_path_planner = new Global_Path_Planner(share_data_initviews, init_views, init_view_ids, first_init_view_id);
		init_dis = gloabl_path_planner->solve();
		init_view_ids = gloabl_path_planner->get_path_id_set();
		//反转路径
		reverse(init_view_ids.begin(), init_view_ids.end());
		cout << "init_dis: " << init_dis << endl;
		delete gloabl_path_planner;
	}
	//输出起始路径
	ofstream fout_move_first(share_data_initviews->pre_path + "/init_movement_v" + to_string(init_view_ids.size()) + ".txt");
	fout_move_first << "init_dis: " << endl;
	fout_move_first << init_dis  << endl;
	fout_move_first << "path: " << endl;
	for (int i = 0; i < init_view_ids.size(); i++) {
		fout_move_first << init_view_ids[i] << '\t';
	}
	fout_move_first << endl;
	fout_move_first.close();
    //go to init_view_ids[0] in ros
    if (visitJoints(init_view_space_pose[init_view_ids[0]])){
        ROS_INFO("MoveitClient: Arm moved to init view pose");
    }
    else{
        ROS_ERROR("MoveitClient: Failed to move arm to init view pose");
    }
    ros::Duration(1.0).sleep();

    num_of_movable_views++;
    share_data->now_camera_pose_world = init_views[init_view_ids[0]].pose_fixed;

    // //获取RGB图像并保存
    // getRGBImage();
    // while(is_rgb_image_received_ == false){
    //     ros::Duration(0.2).sleep();
    // }
    // is_rgb_image_received_ = false;
    // cv::Mat rgb_image = share_data->rgb_now;
    // share_data->access_directory(share_data->save_path + "/images");
    // cv::imwrite(share_data->save_path + "/images/init_rgb_" + to_string(init_view_ids[0]) + ".png", rgb_image);
    //显示视点和路径
    Eigen::Vector4d X(0.05, 0, 0, 1);
    Eigen::Vector4d Y(0, 0.05, 0, 1);
    Eigen::Vector4d Z(0, 0, 0.05, 1);
    Eigen::Vector4d O(0, 0, 0, 1);
    X = share_data->now_camera_pose_world * X;
    Y = share_data->now_camera_pose_world * Y;
    Z = share_data->now_camera_pose_world * Z;
    O = share_data->now_camera_pose_world * O;
    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(-1));
    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(-1));
    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(-1));
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(-1));
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(-1));
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(-1));
    viewer->spinOnce(100);
    Eigen::Affine3d pose_eigen;
    pose_eigen.matrix() = share_data->now_camera_pose_world.matrix();
    geometry_msgs::PoseStamped pose_msg;
    tf::poseEigenToMsg(pose_eigen, pose_msg.pose);
    pose_array_.poses.push_back(pose_msg.pose);
    pose_array_.header.frame_id = "base_link";
    pose_array_.header.stamp = ros::Time::now();

    pose_msg.header.frame_id = "base_link";
    pose_msg.header.stamp = ros::Time::now();
    path_.poses.push_back(pose_msg);

    pose_array_pub_.publish(pose_array_);

	//move follow the path in ros
	for (int i = 1; i < init_view_ids.size(); i++) {
        // Motion Planning, waypoints to move to next best view
        Eigen::Vector3d now_view_xyz(share_data->now_camera_pose_world(0, 3), share_data->now_camera_pose_world(1, 3), share_data->now_camera_pose_world(2, 3));
        Eigen::Vector3d next_view_xyz = init_views[init_view_ids[i]].init_pos;
        vector<Eigen::Vector3d> points;
        int num_of_path = get_trajectory_xyz(points, now_view_xyz, next_view_xyz, share_data->object_center_world, share_data->predicted_size, share_data->move_dis_pre_point, 0.0);
        if (num_of_path == -1) {
            cout << "no path. throw" << endl;
            return;
        }
        if (num_of_path == -2) cout << "Line" << endl;
        if (num_of_path > 0)  cout << "Obstcale" << endl;
        cout << "num_of_path:" << points.size() << endl;

        bool is_global_path = false;

        if(is_global_path) viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(now_view_xyz(0), now_view_xyz(1), now_view_xyz(2)), pcl::PointXYZ(points[0](0), points[0](1), points[0](2)), 128, 0, 128, "trajectory_init" + to_string(i)  + to_string(-1));
        else viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(now_view_xyz(0), now_view_xyz(1), now_view_xyz(2)), pcl::PointXYZ(points[0](0), points[0](1), points[0](2)), 0, 128, 128, "trajectory_init" + to_string(i) + to_string(-1));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory_init"  + to_string(i) + to_string(-1));
        for (int k = 0; k < points.size() - 1; k++) {
            if(is_global_path) viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(points[k](0), points[k](1), points[k](2)), pcl::PointXYZ(points[k + 1](0), points[k + 1](1), points[k + 1](2)), 128, 0, 128, "trajectory_init" + to_string(i) + to_string(k)); 
            else viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(points[k](0), points[k](1), points[k](2)), pcl::PointXYZ(points[k + 1](0), points[k + 1](1), points[k + 1](2)), 0, 128, 128, "trajectory_init" + to_string(i) + to_string(k)); 
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory_init" + to_string(i) + to_string(k));
        }
        viewer->spinOnce(100);

        if(is_global_path)
        {
            Eigen::Affine3d pose_eigen;
            pose_eigen.matrix() = share_data->now_camera_pose_world;
            geometry_msgs::PoseStamped pose_msg;
            tf::poseEigenToMsg(pose_eigen, pose_msg.pose);
            pose_msg.header.frame_id = "base_link";
            pose_msg.header.stamp = ros::Time::now();
            global_path_.poses.push_back(pose_msg);
        }
        
        // get waypoints
        vector<vector<float>> now_waypoints;
        Eigen::Matrix4d now_camera_pose_world = share_data->now_camera_pose_world;
        for(int j=0;j<points.size();j++){
            View temp_view(points[j]);
            temp_view.get_next_camera_pos(now_camera_pose_world,  share_data->object_center_world);
            Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

            cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
            if(j==points.size()-1){
                Eigen::Vector4d X(0.05, 0, 0, 1);
                Eigen::Vector4d Y(0, 0.05, 0, 1);
                Eigen::Vector4d Z(0, 0, 0.05, 1);
                Eigen::Vector4d O(0, 0, 0, 1);
                X = now_camera_pose_world * temp_view.pose.inverse() * X;
                Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                O = now_camera_pose_world * temp_view.pose.inverse() * O;
                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X_init" + to_string(i) + to_string(j));
                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y_init" + to_string(i) + to_string(j));
                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z_init" + to_string(i) + to_string(j));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X_init" + to_string(i) + to_string(j));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y_init" + to_string(i) + to_string(j));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z_init" + to_string(i) + to_string(j));
                viewer->spinOnce(100);
            }

            Eigen::Affine3d pose_eigen;
            pose_eigen.matrix() = now_camera_pose_world * temp_view.pose.inverse().matrix();
            geometry_msgs::PoseStamped pose_msg;
            tf::poseEigenToMsg(pose_eigen, pose_msg.pose);
            pose_msg.header.frame_id = "base_link";
            pose_msg.header.stamp = ros::Time::now();
            if(is_global_path)
                global_path_.poses.push_back(pose_msg);
            else
                path_.poses.push_back(pose_msg);
                
            if (j==points.size()-1)
            {
                pose_array_.poses.push_back(pose_msg.pose);
                pose_array_.header.frame_id = "base_link";
                pose_array_.header.stamp = ros::Time::now();
                pose_array_pub_.publish(pose_array_);
                if(is_global_path)
                {
                    global_path_.header.frame_id = "base_link";
                    global_path_.header.stamp = ros::Time::now();
                    global_path_pub_.publish(global_path_);
                }
                else
                {
                    path_.header.frame_id = "base_link";
                    path_.header.stamp = ros::Time::now();
                    path_pub_.publish(path_);
                }
                
            }

            Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
            Eigen::Quaterniond temp_q(temp_rotation);
            vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
            now_waypoints.push_back(temp_waypoint);

            now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
        }

        cout << "now_waypoints.size():" << now_waypoints.size() << endl;
        // move arm to waypoints
        std::vector<geometry_msgs::Pose> waypoints_msg;
        generateWaypoints(now_waypoints, waypoints_msg);
        if (visitWaypoints(waypoints_msg)){
            num_of_movable_views++;
            ROS_INFO("MoveitClient: Arm moved to original waypoints 1st");
        }
        else{
            // try z table height path movement
            ROS_ERROR("MoveitClient: Failed to move arm to original waypoints 1st");
            
            now_camera_pose_world = share_data->now_camera_pose_world;

            vector<vector<float>> now_waypoints;
            for(int j=0;j<points.size();j++){
                View temp_view(points[j]);
                Eigen::Vector3d now_object_center_world = share_data->object_center_world;
                now_object_center_world(2) = share_data->min_z_table;
                temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

                cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
                if(j==points.size()-1){
                    Eigen::Vector4d X(0.05, 0, 0, 1);
                    Eigen::Vector4d Y(0, 0.05, 0, 1);
                    Eigen::Vector4d Z(0, 0, 0.05, 1);
                    Eigen::Vector4d O(0, 0, 0, 1);
                    X = now_camera_pose_world * temp_view.pose.inverse() * X;
                    Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                    Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                    O = now_camera_pose_world * temp_view.pose.inverse() * O;

                    viewer->removeCorrespondences("X_init" + to_string(i) + to_string(j));
                    viewer->removeCorrespondences("Y_init" + to_string(i) + to_string(j));
                    viewer->removeCorrespondences("Z_init" + to_string(i) + to_string(j));

                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X_init" + to_string(i)+  to_string(j));
                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y_init" + to_string(i) + to_string(j));
                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z_init" + to_string(i) + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X_init" + to_string(i) + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y_init" + to_string(i) + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z_init" + to_string(i) + to_string(j));
                    viewer->spinOnce(100);
                }

                Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                Eigen::Quaterniond temp_q(temp_rotation);
                vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
                now_waypoints.push_back(temp_waypoint);

                now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
            }

            std::vector<geometry_msgs::Pose> waypoints_msg;
            generateWaypoints(now_waypoints, waypoints_msg);
            if (visitWaypoints(waypoints_msg)){
                num_of_movable_views++;
                ROS_INFO("MoveitClient: Arm moved to look at table height waypoints 2nd");
            }
            else{
                ROS_ERROR("MoveitClient: Failed to move arm to look at table height waypoints 2nd");

                if (visitJoints(init_view_space_pose[init_view_ids[i]])){
                    num_of_movable_views++;
                    ROS_INFO("MoveitClient: Arm moved to target look at table height pose 3rd");

                    //now camera pose
                    now_camera_pose_world = init_views[init_view_ids[i]].pose_fixed;
                }
                else{
                    ROS_ERROR("MoveitClient: Failed to move arm to look at table height pose 3rd");

                }  
            }
        }

        geometry_msgs::PoseStamped current_pose = getCurrentPose();

        //更新share_data->now_camera_pose_world
        double now_x = current_pose.pose.position.x;
        double now_y = current_pose.pose.position.y;
        double now_z = current_pose.pose.position.z;
        double now_qx = current_pose.pose.orientation.x;
        double now_qy = current_pose.pose.orientation.y;
        double now_qz = current_pose.pose.orientation.z;
        double now_qw = current_pose.pose.orientation.w;

        Eigen::Quaterniond now_q(now_qw, now_qx, now_qy, now_qz);
        Eigen::Matrix3d now_rotation = now_q.toRotationMatrix();
        Eigen::Matrix4d now_current_camera_pose_world = Eigen::Matrix4d::Identity();
        now_current_camera_pose_world.block<3,3>(0, 0) = now_rotation;
        now_current_camera_pose_world(0, 3) = now_x;
        now_current_camera_pose_world(1, 3) = now_y;
        now_current_camera_pose_world(2, 3) = now_z;

        share_data->now_camera_pose_world = now_current_camera_pose_world * share_data->camera_depth_to_rgb.inverse();

        ros::Duration(1.0).sleep();
        // get RGB image
        getRGBImage();
        while(is_rgb_image_received_ == false){
            ros::Duration(0.2).sleep();
        }
        is_rgb_image_received_ = false;
        cv::Mat rgb_image = share_data->rgb_now;
        share_data->access_directory(share_data->save_path + "/images");
        cv::imwrite(share_data->save_path + "/images/init_rgb_" + to_string(init_view_ids[i]) + ".png", rgb_image);

	}
	//数据区清空
	share_data_initviews.reset();
	view_space_initviews.reset();

    //start planner
    shared_ptr<NBV_Planner> nbv_planner = make_shared<NBV_Planner>(share_data, init_views, first_init_view_id);

    //read finded view poses from file
    ifstream fin_viewspace_pose(share_data->viewspace_path + share_data->name_of_pcd + "/" + to_string(share_data->num_of_views) + "_vs_pose.txt");
    ifstream fin_viewspace_xyzq(share_data->viewspace_path + share_data->name_of_pcd + "/" + to_string(share_data->num_of_views) + "_vs_xyzq.txt");
    vector<vector<double>> view_space_pose;
    vector<vector<double>> view_space_xyzq;
    view_space_pose.resize(share_data->num_of_views);
    view_space_xyzq.resize(share_data->num_of_views);
    for(int i = 0; i < share_data->num_of_views; i++){
        int view_id;
        fin_viewspace_pose >> view_id;
        view_space_pose[view_id].resize(6);
        fin_viewspace_pose>>view_space_pose[view_id][0]>>view_space_pose[view_id][1]>>view_space_pose[view_id][2]>>view_space_pose[view_id][3]>>view_space_pose[view_id][4]>>view_space_pose[view_id][5];
        fin_viewspace_xyzq >> view_id;
        view_space_xyzq[view_id].resize(7);
        fin_viewspace_xyzq>>view_space_xyzq[view_id][0]>>view_space_xyzq[view_id][1]>>view_space_xyzq[view_id][2]>>view_space_xyzq[view_id][3]>>view_space_xyzq[view_id][4]>>view_space_xyzq[view_id][5]>>view_space_xyzq[view_id][6];
        //covnert to Eigen::Matrix4d
        double start_x = view_space_xyzq[view_id][0];
        double start_y = view_space_xyzq[view_id][1];
        double start_z = view_space_xyzq[view_id][2];
        double start_qx = view_space_xyzq[view_id][3];
        double start_qy = view_space_xyzq[view_id][4];
        double start_qz = view_space_xyzq[view_id][5];
        double start_qw = view_space_xyzq[view_id][6];
        Eigen::Quaterniond start_q(start_qw, start_qx, start_qy, start_qz);
        Eigen::Matrix3d start_rotation = start_q.toRotationMatrix();
        Eigen::Matrix4d start_pose_world = Eigen::Matrix4d::Identity();
        start_pose_world.block<3,3>(0, 0) = start_rotation;
        start_pose_world(0, 3) = start_x;
        start_pose_world(1, 3) = start_y;
        start_pose_world(2, 3) = start_z;
        Eigen::Matrix4d view_pose_world = start_pose_world * share_data->camera_depth_to_rgb.inverse();
        nbv_planner->view_space->views[view_id].pose_fixed = view_pose_world;
    }

	cout << "start view planning." << endl;
	while (nbv_planner->loop_once()){
		int nbv_id = nbv_planner->get_nbv_id();
		cout << "next_view_id: " << nbv_id << endl;

        if(share_data->method_of_IG == PVBCoverage){
            //read finded view poses from file again
            ifstream fin_viewspace_pose(share_data->viewspace_path + share_data->name_of_pcd + "/" + to_string(share_data->num_of_views) + "_vs_pose.txt");
            view_space_pose.clear();
            view_space_pose.resize(share_data->num_of_views);
            ifstream fin_viewspace_xyzq(share_data->viewspace_path + share_data->name_of_pcd + "/" + to_string(share_data->num_of_views) + "_vs_xyzq.txt");
            view_space_xyzq.clear();
            view_space_xyzq.resize(share_data->num_of_views);
            for(int i = 0; i < share_data->num_of_views; i++){
                int view_id;
                fin_viewspace_pose >> view_id;
                view_space_pose[view_id].resize(7);
                fin_viewspace_pose>>view_space_pose[view_id][0]>>view_space_pose[view_id][1]>>view_space_pose[view_id][2]>>view_space_pose[view_id][3]>>view_space_pose[view_id][4]>>view_space_pose[view_id][5];
                fin_viewspace_xyzq >> view_id;
                view_space_xyzq[view_id].resize(7);
                fin_viewspace_xyzq>>view_space_xyzq[view_id][0]>>view_space_xyzq[view_id][1]>>view_space_xyzq[view_id][2]>>view_space_xyzq[view_id][3]>>view_space_xyzq[view_id][4]>>view_space_xyzq[view_id][5]>>view_space_xyzq[view_id][6];
                //covnert to Eigen::Matrix4d
                double start_x = view_space_xyzq[view_id][0];
                double start_y = view_space_xyzq[view_id][1];
                double start_z = view_space_xyzq[view_id][2];
                double start_qx = view_space_xyzq[view_id][3];
                double start_qy = view_space_xyzq[view_id][4];
                double start_qz = view_space_xyzq[view_id][5];
                double start_qw = view_space_xyzq[view_id][6];
                Eigen::Quaterniond start_q(start_qw, start_qx, start_qy, start_qz);
                Eigen::Matrix3d start_rotation = start_q.toRotationMatrix();
                Eigen::Matrix4d start_pose_world = Eigen::Matrix4d::Identity();
                start_pose_world.block<3,3>(0, 0) = start_rotation;
                start_pose_world(0, 3) = start_x;
                start_pose_world(1, 3) = start_y;
                start_pose_world(2, 3) = start_z;
                Eigen::Matrix4d view_pose_world = start_pose_world * share_data->camera_depth_to_rgb.inverse();
                nbv_planner->view_space->views[view_id].pose_fixed = view_pose_world;
            }
        }
		
        // Motion Planning, waypoints to move to next best view
        Eigen::Vector3d now_view_xyz(share_data->now_camera_pose_world(0, 3), share_data->now_camera_pose_world(1, 3), share_data->now_camera_pose_world(2, 3));
        Eigen::Vector3d next_view_xyz = nbv_planner->view_space->views[nbv_id].init_pos;
        vector<Eigen::Vector3d> points;
        int num_of_path = get_trajectory_xyz(points, now_view_xyz, next_view_xyz, share_data->object_center_world, share_data->predicted_size, share_data->move_dis_pre_point, 0.0);
        if (num_of_path == -1) {
            cout << "no path. throw" << endl;
            return;
        }
        if (num_of_path == -2) cout << "Line" << endl;
        if (num_of_path > 0)  cout << "Obstcale" << endl;
        cout << "num_of_path:" << points.size() << endl;

        bool is_global_path = (share_data->method_of_IG == PVBCoverage || share_data->method_of_IG == RandomOneshot);

        if(is_global_path) viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(now_view_xyz(0), now_view_xyz(1), now_view_xyz(2)), pcl::PointXYZ(points[0](0), points[0](1), points[0](2)), 128, 0, 128, "trajectory" + to_string(nbv_planner->iteration)  + to_string(-1));
        else viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(now_view_xyz(0), now_view_xyz(1), now_view_xyz(2)), pcl::PointXYZ(points[0](0), points[0](1), points[0](2)), 0, 128, 128, "trajectory" + to_string(nbv_planner->iteration) + to_string(-1));
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory"  + to_string(nbv_planner->iteration) + to_string(-1));
        for (int k = 0; k < points.size() - 1; k++) {
            if(is_global_path) viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(points[k](0), points[k](1), points[k](2)), pcl::PointXYZ(points[k + 1](0), points[k + 1](1), points[k + 1](2)), 128, 0, 128, "trajectory" + to_string(nbv_planner->iteration) + to_string(k)); 
            else viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(points[k](0), points[k](1), points[k](2)), pcl::PointXYZ(points[k + 1](0), points[k + 1](1), points[k + 1](2)), 0, 128, 128, "trajectory" + to_string(nbv_planner->iteration) + to_string(k)); 
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "trajectory" + to_string(nbv_planner->iteration) + to_string(k));
        }
        viewer->spinOnce(100);

        if(is_global_path)
        {
            Eigen::Affine3d pose_eigen;
            pose_eigen.matrix() = share_data->now_camera_pose_world;
            geometry_msgs::PoseStamped pose_msg;
            tf::poseEigenToMsg(pose_eigen, pose_msg.pose);
            pose_msg.header.frame_id = "base_link";
            pose_msg.header.stamp = ros::Time::now();
            global_path_.poses.push_back(pose_msg);
        }
        
        // get waypoints
        vector<vector<float>> now_waypoints;
        Eigen::Matrix4d now_camera_pose_world = share_data->now_camera_pose_world;
        for(int j=0;j<points.size();j++){
            View temp_view(points[j]);
            temp_view.get_next_camera_pos(now_camera_pose_world, share_data->object_center_world);
            Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

            cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
            if(j==points.size()-1){
                Eigen::Vector4d X(0.05, 0, 0, 1);
                Eigen::Vector4d Y(0, 0.05, 0, 1);
                Eigen::Vector4d Z(0, 0, 0.05, 1);
                Eigen::Vector4d O(0, 0, 0, 1);
                X = now_camera_pose_world * temp_view.pose.inverse() * X;
                Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                O = now_camera_pose_world * temp_view.pose.inverse() * O;
                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(nbv_planner->iteration) + to_string(j));
                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(nbv_planner->iteration) + to_string(j));
                viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(nbv_planner->iteration) + to_string(j));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(nbv_planner->iteration) + to_string(j));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(nbv_planner->iteration) + to_string(j));
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(nbv_planner->iteration) + to_string(j));
                viewer->spinOnce(100);
            }

            Eigen::Affine3d pose_eigen;
            pose_eigen.matrix() = now_camera_pose_world * temp_view.pose.inverse().matrix();
            geometry_msgs::PoseStamped pose_msg;
            tf::poseEigenToMsg(pose_eigen, pose_msg.pose);
            pose_msg.header.frame_id = "base_link";
            pose_msg.header.stamp = ros::Time::now();
            if(is_global_path)
                global_path_.poses.push_back(pose_msg);
            else
                path_.poses.push_back(pose_msg);
                
            if (j==points.size()-1)
            {
                pose_array_.poses.push_back(pose_msg.pose);
                pose_array_.header.frame_id = "base_link";
                pose_array_.header.stamp = ros::Time::now();
                pose_array_pub_.publish(pose_array_);
                if(is_global_path)
                {
                    global_path_.header.frame_id = "base_link";
                    global_path_.header.stamp = ros::Time::now();
                    global_path_pub_.publish(global_path_);
                }
                else
                {
                    path_.header.frame_id = "base_link";
                    path_.header.stamp = ros::Time::now();
                    path_pub_.publish(path_);
                }
                
            }

            Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
            Eigen::Quaterniond temp_q(temp_rotation);
            vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
            now_waypoints.push_back(temp_waypoint);

            now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
        }
        // move arm to waypoints
        std::vector<geometry_msgs::Pose> waypoints_msg;
        generateWaypoints(now_waypoints, waypoints_msg);
        if (visitWaypoints(waypoints_msg)){
            num_of_movable_views++;
            ROS_INFO("MoveitClient: Arm moved to original waypoints 1st");
        }
        else{
            // try z table height path movement
            ROS_ERROR("MoveitClient: Failed to move arm to original waypoints 1st");
            
            now_camera_pose_world = share_data->now_camera_pose_world;

            vector<vector<float>> now_waypoints;
            for(int j=0;j<points.size();j++){
                View temp_view(points[j]);
                Eigen::Vector3d now_object_center_world = share_data->object_center_world;
                now_object_center_world(2) = share_data->min_z_table;
                temp_view.get_next_camera_pos(now_camera_pose_world,  now_object_center_world);
                Eigen::Matrix4d temp_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse() * share_data->camera_depth_to_rgb; 

                cout<<"temp_camera_pose_world:"<<temp_camera_pose_world<<endl;
                if(j==points.size()-1){
                    Eigen::Vector4d X(0.05, 0, 0, 1);
                    Eigen::Vector4d Y(0, 0.05, 0, 1);
                    Eigen::Vector4d Z(0, 0, 0.05, 1);
                    Eigen::Vector4d O(0, 0, 0, 1);
                    X = now_camera_pose_world * temp_view.pose.inverse() * X;
                    Y = now_camera_pose_world * temp_view.pose.inverse() * Y;
                    Z = now_camera_pose_world * temp_view.pose.inverse() * Z;
                    O = now_camera_pose_world * temp_view.pose.inverse() * O;

                    viewer->removeCorrespondences("X" + to_string(nbv_planner->iteration) + to_string(j));
                    viewer->removeCorrespondences("Y" + to_string(nbv_planner->iteration) + to_string(j));
                    viewer->removeCorrespondences("Z" + to_string(nbv_planner->iteration) + to_string(j));

                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(X(0), X(1), X(2)), 255, 0, 0, "X" + to_string(nbv_planner->iteration)+  to_string(j));
                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Y(0), Y(1), Y(2)), 0, 255, 0, "Y" + to_string(nbv_planner->iteration) + to_string(j));
                    viewer->addLine<pcl::PointXYZ>(pcl::PointXYZ(O(0), O(1), O(2)), pcl::PointXYZ(Z(0), Z(1), Z(2)), 0, 0, 255, "Z" + to_string(nbv_planner->iteration) + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "X" + to_string(nbv_planner->iteration) + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Y" + to_string(nbv_planner->iteration) + to_string(j));
                    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "Z" + to_string(nbv_planner->iteration) + to_string(j));
                    viewer->spinOnce(100);
                }

                Eigen::Matrix3d temp_rotation = temp_camera_pose_world.block<3,3>(0, 0);
                Eigen::Quaterniond temp_q(temp_rotation);
                vector<float> temp_waypoint = { float(temp_camera_pose_world(0, 3)), float(temp_camera_pose_world(1, 3)), float(temp_camera_pose_world(2, 3)), float(temp_q.x()), float(temp_q.y()), float(temp_q.z()), float(temp_q.w()) };
                now_waypoints.push_back(temp_waypoint);

                now_camera_pose_world = now_camera_pose_world * temp_view.pose.inverse();
            }

            std::vector<geometry_msgs::Pose> waypoints_msg;
            generateWaypoints(now_waypoints, waypoints_msg);
            if (visitWaypoints(waypoints_msg)){
                num_of_movable_views++;
                ROS_INFO("MoveitClient: Arm moved to look at table height waypoints 2nd");
            }
            else{
                ROS_ERROR("MoveitClient: Failed to move arm to look at table height waypoints 2nd");

                if (visitJoints(view_space_pose[nbv_id])){
                    num_of_movable_views++;
                    ROS_INFO("MoveitClient: Arm moved to target look at table height pose 3rd");

                    //now camera pose
                    now_camera_pose_world = nbv_planner->view_space->views[nbv_id].pose_fixed;

                }
                else{
                    ROS_ERROR("MoveitClient: Failed to move arm to look at table height pose 3rd");

                }  
            }

        }

        share_data->now_camera_pose_world = now_camera_pose_world;

        geometry_msgs::PoseStamped current_pose = getCurrentPose();
        ros::Duration(1.0).sleep();

        //获取RGB图像并保存
        getRGBImage();
        while(is_rgb_image_received_ == false){
            ros::Duration(0.2).sleep();
        }
        is_rgb_image_received_ = false;
        cv::Mat rgb_image = share_data->rgb_now;
        share_data->access_directory(share_data->save_path + "/images");
        cv::imwrite(share_data->save_path + "/images/plan_rgb_" + to_string(nbv_id) + ".png", rgb_image);

	}

    cout<<"num_of_iterations: "<<nbv_planner->iteration<<endl;
    cout<<"num_of_movable_views: "<<num_of_movable_views<<endl;

    double final_time_stamp = ros::Time::now().toSec();

    cout<<"reconstruction mode finish"<<endl;

    ofstream fout_time(share_data->save_path + "/runtime.txt");
    fout_time<<final_time_stamp - initial_time_stamp<<endl;

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
    viewer->close();

     //数据区清空
	nbv_planner.reset();
	share_data.reset();

    return;
}

void RosInterface::getPointCloud()
{
    // point cloud subscriber
    sensor_msgs::PointCloud2ConstPtr point_cloud_msg = ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/camera/depth/color/points", ros::Duration(10.0));

    if (point_cloud_msg)
    {
        // pre-process point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc = preProcessPointCloud(point_cloud_msg);

        *share_data->cloud_now = *pc;

        // set flag
        is_point_cloud_received_ = true;
    }
}

void RosInterface::getRGBImage()
{
    // rgb subscriber
    sensor_msgs::ImageConstPtr rgb_msg = ros::topic::waitForMessage<sensor_msgs::Image>("/camera/color/image_raw", ros::Duration(10.0));

    if (rgb_msg)
    {
        // convert to cv::Mat
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("RosInterface: cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat rgb = cv_ptr->image;

        share_data->rgb_now = rgb.clone();

        // set flag
        is_rgb_image_received_ = true;
    }
}

void RosInterface::initMoveitClient()
{
    move_group_arm_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(PLANNING_GROUP_ARM);

    // move to home position
    //if (moveArmToHome())
    //{
    //    ROS_INFO("MoveitClient: Arm moved to home position");
    //}
    //else
    //{
    //    ROS_ERROR("MoveitClient: Failed to move arm to home position");
    //}

    // print current pose
    // auto _ = getCurrentPose();
}

bool RosInterface::moveArmToHome()
{
    // move to home position
    // 0.011035, 0.390509, 0.338996, -0.542172, 0.472681, 0.521534, 0.458939
     geometry_msgs::Pose home_pose;

    // old home pose
    //1.219672, -1.815668, 1.872075, 4.685153, -1.596196, 4.232074

    //1.50342 -1.67972 1.52916 -1.40414 -1.64011 -0.0491012 
    vector<double> home_joints = { 1.50342, -1.67972, 1.52916, -1.40414, -1.64011, -0.0491012 };

    // move arm to waypoints
    if (visitJoints(home_joints)){
        ROS_INFO("MoveitClient: Arm moved to home");
        return true;
    }
    else{
        ROS_ERROR("MoveitClient: Failed to move arm to home");
        return false;
    }
}

bool RosInterface::visitPose(const geometry_msgs::Pose& pose)
{
    move_group_arm_->setJointValueTarget(pose);

    bool success = (move_group_arm_->plan(arm_plan_) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success)
    {
        move_group_arm_->execute(arm_plan_);
        ROS_INFO("MoveitClient: Arm moved to the position");
        return true;
    }
    else
    {
        ROS_ERROR("MoveitClient: Failed to move arm to the position");
        return false;
    }
}

bool RosInterface::visitJoints(const std::vector<double>& joints)
{
    move_group_arm_->setJointValueTarget(joints);

    bool success = (move_group_arm_->plan(arm_plan_) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if (success)
    {
        move_group_arm_->execute(arm_plan_);
        ROS_INFO("MoveitClient: Arm moved to the position");
        return true;
    }
    else
    {
        ROS_ERROR("MoveitClient: Failed to move arm to the position");
        return false;
    }
}

bool RosInterface::visitWaypoints(const std::vector<geometry_msgs::Pose>& waypoints, float jump_threshold)
{
    // move arm in a cartesian path
    moveit_msgs::RobotTrajectory trajectory;
    double fraction = move_group_arm_->computeCartesianPath(waypoints, 0.01, jump_threshold, trajectory);

    int maxtries = 100;
    int attempts = 0;
    while (fraction < 1.0 && attempts < maxtries)
    {
        fraction = move_group_arm_->computeCartesianPath(waypoints, 0.01, jump_threshold, trajectory);
        attempts++;
        
        if(attempts % 30 == 0){
            ROS_INFO("MoveitClient: Cartesian path computed with fraction: %f ", fraction);
            ROS_INFO("MoveitClient: Retrying to compute cartesian path");
        }
    }

    ROS_INFO("MoveitClient: Cartesian path computed with fraction: %f ", fraction);

    if (fraction < 1.0)
    {
        ROS_ERROR("MoveitClient: Failed to compute cartesian path");
        return false;
    }

    // execute trajectory
    moveit::core::MoveItErrorCode result = move_group_arm_->execute(trajectory);
    if (result != moveit::core::MoveItErrorCode::SUCCESS)
    {
        ROS_ERROR("MoveitClient: Failed to execute trajectory");
        return false;
    }

    return true;
}

void RosInterface::generateWaypoints(const std::vector<std::vector<float>>& waypoints, std::vector<geometry_msgs::Pose>& waypoints_msg)
{
    for (const auto& waypoint : waypoints)
    {
        geometry_msgs::Pose waypoint_msg;
        waypoint_msg.position.x = waypoint[0];
        waypoint_msg.position.y = waypoint[1];
        waypoint_msg.position.z = waypoint[2];

        waypoint_msg.orientation.x = waypoint[3];
        waypoint_msg.orientation.y = waypoint[4];
        waypoint_msg.orientation.z = waypoint[5];
        waypoint_msg.orientation.w = waypoint[6];

        waypoints_msg.push_back(waypoint_msg);
    }
}

geometry_msgs::PoseStamped RosInterface::getCurrentPose()
{
    geometry_msgs::PoseStamped current_pose = move_group_arm_->getCurrentPose();
    ROS_INFO("MoveitClient: Current pose (xyz): %f, %f, %f", current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z);

    // convert quaternion to euler angles
    tf2::Quaternion q(current_pose.pose.orientation.x,
                     current_pose.pose.orientation.y,
                     current_pose.pose.orientation.z,
                     current_pose.pose.orientation.w);

    ROS_INFO("MoveitClient: Current pose (q): %f, %f, %f, %f", q.x(), q.y(), q.z(), q.w());
    
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);

    // convert to degrees
    roll = roll * 180 / M_PI;
    pitch = pitch * 180 / M_PI;
    yaw = yaw * 180 / M_PI;

    ROS_INFO("MoveitClient: Current pose (rpy): %f, %f, %f", roll, pitch, yaw);

    return current_pose;
}

std::vector<double> RosInterface::getJointValues()
{
    std::vector<double> joint_values = move_group_arm_->getCurrentJointValues();

    ROS_INFO("MoveitClient: Current joints: %f, %f, %f, %f, %f, %f", joint_values[0], joint_values[1], joint_values[2], joint_values[3], joint_values[4], joint_values[5]);

    return joint_values;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ma_scvp_real");
    ros::NodeHandle nh;
    ros::AsyncSpinner spinner(4);
    spinner.start();
    RosInterface ros_interface(nh);
    ros_interface.run();
    ros::waitForShutdown();
    return 0;
}