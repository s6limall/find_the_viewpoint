//
// Created by ayush on 5/21/24.
//

#ifndef PERCEPTION_HPP
#define PERCEPTION_HPP

#include <string>
#include <memory>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/PolygonMesh.h>
#include <Eigen/Core>

class Perception {
public:
    struct CameraConfig {
        int width, height; // Image dimensions
        float fov_x, fov_y; // Horizontal and vertical field of view in radians
        Eigen::Matrix3f intrinsics; // Intrinsic matrix of the camera
    };

    // Constructor to initialize Perception with a given object path.
    explicit Perception(const std::string &object_path);

    ~Perception();

    // Configure the camera with specified width, height, and field of view.
    void configureCamera(int width, int height, float fov_x, float fov_y);

    // Render the object using the given camera pose and save the image.
    void render(const Eigen::Matrix4f &camera_pose, const std::string &image_save_path) const;

private:
    CameraConfig config; // Configuration for the camera
    std::shared_ptr<pcl::visualization::PCLVisualizer> viewer; // Visualizer for rendering the mesh
    pcl::PolygonMesh::Ptr mesh_ply; // Polygon mesh loaded from a file

    // Load a polygon mesh from a PLY file.
    void loadMesh(const std::string &object_path);

    // Configure the PCL visualizer.
    void setupViewer();

    // Normalize the mesh to fit within a unit cube at the origin.
    void normalizeObject();

    // Update the intrinsic matrix based on camera parameters.
    void updateIntrinsics(int width, int height, float fov_x, float fov_y);

    // Center the point cloud at the origin.
    void centerPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const Eigen::Vector4f &centroid);

    // Scale the point cloud to fit within a unit cube.
    void scalePointCloudToFitUnitCube(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
};

#endif // PERCEPTION_HPP
