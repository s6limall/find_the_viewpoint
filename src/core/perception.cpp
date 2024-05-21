//
// Created by ayush on 5/21/24.
//

#include "../../include/core/perception.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/io/io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/conversions.h>

using namespace std;

Perception::Perception(const string &object_path) {
    width = 640;
    height = 480;
    fov_x = 0.95;
    fov_y = 0.75;
    double fx = width / (2 * tan(fov_x / 2));
    double fy = height / (2 * tan(fov_y / 2));
    intrinsics << fx, 0, width / 2,
                  0, fy, height / 2,
                  0, 0, 1;

    mesh_ply = pcl::PolygonMesh::Ptr(new pcl::PolygonMesh);
    pcl::io::loadPolygonFilePLY(object_path, *mesh_ply);
    if (mesh_ply->cloud.data.empty() || mesh_ply->polygons.empty()) {
        cout << "Load object: " << object_path << " failed!" << endl;
        exit(1);
    }

    normalizeObject();

    viewer = pcl::visualization::PCLVisualizer::Ptr(new pcl::visualization::PCLVisualizer("Render Viewer"));
    viewer->setBackgroundColor(255, 255, 255);
    viewer->initCameraParameters();
    viewer->addPolygonMesh(*mesh_ply, "object");
    viewer->setSize(width, height);
    viewer->spinOnce(100);
}

Perception::~Perception() {
    viewer->removePolygonMesh("mesh_ply");
    viewer->close();
}

void Perception::normalizeObject() {
    int mesh_data_offset = mesh_ply->cloud.data.size() / mesh_ply->cloud.width / mesh_ply->cloud.height;
    pcl::PointCloud<pcl::PointXYZ>::Ptr vertex;
    vertex = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(mesh_ply->cloud, *vertex);
    vector<Eigen::Vector3d> points;
    for (auto& ptr : vertex->points) {
        points.push_back(Eigen::Vector3d(ptr.x, ptr.y, ptr.z));
    }
    Eigen::Vector3d object_center = Eigen::Vector3d::Zero();
    for (auto& ptr : points) {
        object_center += ptr;
    }
    object_center /= points.size();
    double object_size = 0.0;
    for (auto& ptr : points) {
        object_size = max(object_size, (object_center - ptr).norm());
    }
    double scale = 1.0 / object_size;
    for (int i = 0; i < mesh_ply->cloud.data.size(); i += mesh_data_offset) {
        int arrayPosX = i + mesh_ply->cloud.fields[0].offset;
        int arrayPosY = i + mesh_ply->cloud.fields[1].offset;
        int arrayPosZ = i + mesh_ply->cloud.fields[2].offset;
        float X, Y, Z;
        memcpy(&X, &mesh_ply->cloud.data[arrayPosX], sizeof(float));
        memcpy(&Y, &mesh_ply->cloud.data[arrayPosY], sizeof(float));
        memcpy(&Z, &mesh_ply->cloud.data[arrayPosZ], sizeof(float));
        X = float((X - object_center(0)) * scale);
        Y = float((Y - object_center(1)) * scale);
        Z = float((Z - object_center(2)) * scale);
        memcpy(&mesh_ply->cloud.data[arrayPosX], &X, sizeof(float));
        memcpy(&mesh_ply->cloud.data[arrayPosY], &Y, sizeof(float));
        memcpy(&mesh_ply->cloud.data[arrayPosZ], &Z, sizeof(float));
    }
}

void Perception::render(View view, const string &image_save_path) {
    Eigen::Matrix4f extrinsics = view.pose_6d.cast<float>();
    viewer->setCameraParameters(intrinsics, extrinsics);
    viewer->spinOnce(100);

    string test_image_save_path = image_save_path.substr(0, image_save_path.size() - 4) + "_test.png";
    viewer->saveScreenshot(test_image_save_path);

    cv::Mat img = cv::imread(test_image_save_path);
    if (img.cols != width || img.rows != height) {
        img = img(cv::Rect(img.cols - width, img.rows - height, width, height));
    }
    cv::imwrite(image_save_path, img);
    remove(test_image_save_path.c_str());
}
