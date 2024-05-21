//
// Created by ayush on 5/21/24.
//

#ifndef PERCEPTION_HPP
#define PERCEPTION_HPP

#include <string>
#include <vector>
#include <Eigen/Core>
#include <pcl/PolygonMesh.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "view.hpp"

class Perception {
public:
    double width;
    double height;
    double fov_x;
    double fov_y;
    Eigen::Matrix3f intrinsics;
    pcl::PolygonMesh::Ptr mesh_ply;
    pcl::visualization::PCLVisualizer::Ptr viewer;

    Perception(const std::string &object_path);
    ~Perception();

    void render(View view, const std::string &image_save_path = "../rgb.png");

private:
    void normalizeObject();
};

#endif // PERCEPTION_HPP
