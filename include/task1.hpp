#ifndef VIEW_PLANNING_SIMULATOR_HPP
#define VIEW_PLANNING_SIMULATOR_HPP

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/common/io.h>
#include <pcl/geometry/polygon_mesh.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/conversions.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "../include/config.hpp"
#include "../include/image.hpp"

typedef unsigned long long pop_t;

using namespace std;

// Robot class
class Robot {
    // Placeholder for Robot class implementation
};

// View class
class View {
public:
    Eigen::Matrix4d pose_6d; // 6D pose of the camera

    // Constructor
    View();

    // Destructor
    ~View();

    // Get the 6D pose of the camera
    void compute_pose_from_positon_and_object_center(Eigen::Vector3d positon, Eigen::Vector3d object_center);
};

// Perception class
class Perception {
public:
    double width; // Width of the camera
    double height; // Height of the camera
    double fov_x; // Field of view in x direction
    double fov_y; // Field of view in y direction
    Eigen::Matrix3f intrinsics; // Camera intrinsics

    pcl::PolygonMesh::Ptr mesh_ply; // Object mesh

    pcl::visualization::PCLVisualizer::Ptr viewer; // Viewer

    // Constructor
    Perception(string object_path);

    // Destructor
    ~Perception();

    // Render RGB image from a viewpoint
    void render(View view, string image_save_path = "../rgb.png");
};

// View_Planning_Simulator class
class View_Planning_Simulator {
public:
    Perception* perception_simulator; // perception simulator
    cv::Mat target_image; // target image
    vector<View> view_space; // view space
    vector<View> selected_views; // selected views
    vector<cv::Mat> rendered_images; // rendered images

    // Constructor
    View_Planning_Simulator(Perception* _perception_simulator, cv::Mat _target_image, vector<View> _view_space = vector<View>());

    // Destructor
    ~View_Planning_Simulator();

    // Render view image
    cv::Mat render_view_image(View view);

    // Check if the view is target
    bool is_target(View view);

    // Search for the next view
    View search_next_view();

    // Search the best view until find the target
    void loop();
};

// Task1 function
void run_level_1();
void run_level_2();

// Main function
int main();

#endif // VIEW_PLANNING_SIMULATOR_HPP
