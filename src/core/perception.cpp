//
// Created by ayush on 5/21/24.
//

#include "../../include/core/perception.hpp"
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/io/vtk_lib_io.h>

using namespace std;

Perception::Perception(const std::string &object_path)
    : viewer(std::make_shared<pcl::visualization::PCLVisualizer>("Viewer")),
      mesh_ply(std::make_shared<pcl::PolygonMesh>()) {
    loadMesh(object_path); // Load the 3D model mesh from a file.
    setupViewer(); // Setup the PCL visualizer.
    normalizeObject(); // Normalize the mesh to a unit cube centered at the origin.
}

Perception::~Perception() {
    viewer->close(); // Close the viewer on destruction to free resources.
}

// Configures camera parameters and updates the intrinsic matrix.
void Perception::configureCamera(int width, int height, float fov_x, float fov_y) {
    updateIntrinsics(width, height, fov_x, fov_y); // Update the intrinsic matrix based on new camera parameters.
}

// Render the object using the given camera pose and save the image.
void Perception::render(const Eigen::Matrix4f &camera_pose, const std::string &image_save_path) const {
    viewer->setSize(config.width, config.height); // Set viewer size to camera dimensions

    // Extract the camera position from the pose matrix
    Eigen::Vector3f camera_position = camera_pose.block<3, 1>(0, 3);

    // Calculate the look-at point
    Eigen::Vector3f look_at_point = camera_position + camera_pose.block<3, 1>(0, 2);

    // Extract the up vector from the pose matrix
    Eigen::Vector3f up_vector = camera_pose.block<3, 1>(0, 1);

    // Set the camera parameters in the PCL visualizer
    viewer->setCameraPosition(
        camera_position.x(), camera_position.y(), camera_position.z(),
        look_at_point.x(), look_at_point.y(), look_at_point.z(),
        up_vector.x(), up_vector.y(), up_vector.z()
    );

    // Render the scene
    viewer->spinOnce(100);

    // Capture the screenshot
    viewer->saveScreenshot(image_save_path);
}


// Loads the mesh from a PLY file and throws an error if the file cannot be loaded.
void Perception::loadMesh(const std::string &object_path) {
    if (!pcl::io::loadPolygonFilePLY(object_path, *mesh_ply)) {
        throw std::runtime_error("Failed to load mesh from: " + object_path);
    }
}

// Sets initial parameters for the viewer, such as background color and camera parameters.
void Perception::setupViewer() {
    viewer->setBackgroundColor(0, 0, 0); // Set a black background for better visibility.
    viewer->addPolygonMesh(*mesh_ply, "mesh"); // Add the loaded mesh to the viewer.
    viewer->initCameraParameters(); // Initialize default camera parameters.
    viewer->setSize(config.width, config.height); // Set the size of the viewer window.
}

// Normalizes the mesh to ensure it fits within a unit cube and is centered at the origin.
void Perception::normalizeObject() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromPCLPointCloud2(mesh_ply->cloud, *cloud); // Convert the polygon mesh to a point cloud for processing.

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid); // Calculate the centroid of the cloud.
    centerPointCloud(cloud, centroid); // Translate the cloud so that the centroid is at the origin.

    scalePointCloudToFitUnitCube(cloud); // Scale the cloud so that it fits within a unit cube.
    pcl::toPCLPointCloud2(*cloud, mesh_ply->cloud); // Convert the processed cloud back to the polygon mesh.

    viewer->updatePolygonMesh(*mesh_ply, "mesh"); // Update the mesh in the viewer with the normalized mesh.
}

// Updates the intrinsic matrix based on the specified camera parameters.
void Perception::updateIntrinsics(int width, int height, float fov_x, float fov_y) {
    double fx = width / (2.0 * tan(fov_x / 2.0)); // Calculate the focal length based on the field of view.
    double fy = height / (2.0 * tan(fov_y / 2.0));

    config.intrinsics << fx, 0, width / 2, // Update the intrinsic matrix with new values.
            0, fy, height / 2,
            0, 0, 1;
}

// Centers the point cloud at the origin by adjusting all points relative to the centroid.
void Perception::centerPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const Eigen::Vector4f &centroid) {
    Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
    translation.block<3, 1>(0, 3) = -centroid.head<3>();
    // Create a translation matrix to move the centroid to the origin.
    pcl::transformPointCloud(*cloud, *cloud, translation); // Apply the translation to the cloud.
}

// Scales the point cloud to fit within a unit cube by determining the maximum dimension and scaling all dimensions equally.
void Perception::scalePointCloudToFitUnitCube(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    Eigen::Vector4f min_point, max_point;
    pcl::getMinMax3D(*cloud, min_point, max_point); // Determine the min and max points in the cloud.

    float max_dim = std::max(max_point.x() - min_point.x(),
                             std::max(max_point.y() - min_point.y(), max_point.z() - min_point.z()));
    Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
    scale.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() / max_dim;
    // Calculate a scale matrix to fit the cloud within a unit cube.

    pcl::transformPointCloud(*cloud, *cloud, scale); // Apply the scale transformation to the cloud.
}
