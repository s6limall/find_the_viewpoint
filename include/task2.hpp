#ifndef TASK2_HPP
#define TASK2_HPP

#pragma once
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
#include <opencv2/core/eigen.hpp>

#include "../include/config.hpp"
#include "../include/image.hpp"

typedef unsigned long long pop_t;

using namespace std;

namespace task2{
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

    bool is_test_viewtarget(View src_view, double bst_score);

    // calculate centroid between A,B, and C on a sphere
	View calculate_new_center(const View & A, const View & B, const View & C);

    // Apply H to candiate_view on the plane that is intersecting the view and has the candiate view as the support vector
	View applyHomographyToView(const View & candidate_view, const Eigen::Matrix3d& H);

    // distance
	View fine_registration(const View & candidate_view);

    // Search for the next view
    View dfs_next_view(const View & A, const View & B, const View & C, size_t & bst_score);

    // Search the best view until find the target
    void dfs();
};


void run_level_3();

}

#endif // TASK2_HPP
