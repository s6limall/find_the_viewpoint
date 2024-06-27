// File: core/perception.cpp

#include "core/perception.hpp"
#include "config/configuration.hpp"
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/io/vtk_lib_io.h>

using namespace std;

namespace core {
    Perception::Perception(const std::string &object_path) :
        viewer_(std::make_shared<pcl::visualization::PCLVisualizer>("Viewer")),
        mesh_(std::make_shared<pcl::PolygonMesh>()),
        camera_(std::make_shared<Camera>()) {

        const auto &config = config::Configuration::getInstance();
        auto width = config.get<int>("camera.width", 640);
        auto height = config.get<int>("camera.height", 480);
        auto fov_x = config.get<float>("camera.fov_x", 60.0f);
        auto fov_y = config.get<float>("camera.fov_y", 45.0f);

        // Configure the camera with these settings
        camera_->setIntrinsics(width, height, fov_x, fov_y);

        loadMesh(object_path); // Load the 3D model mesh from a file.
        setupViewer(); // Setup the PCL visualizer.
        normalizeMesh(); // Normalize the mesh to a unit cube centered at the origin.

        spdlog::debug("Perception initialized with camera intrinsics: width={}, height={}, fov_x={}, fov_y={}", width,
                      height, fov_x, fov_y);
    }

    Perception::~Perception() {
        if (!viewer_ || viewer_->wasStopped()) {
            viewer_->close(); // Close the PCL visualizer.
        }

        spdlog::debug("Perception instance destroyed.");
    }

    // Configures camera parameters and updates the intrinsic matrix.
    void Perception::configureCamera(int width, int height, float fov_x, float fov_y) const {
        camera_->setIntrinsics(width, height, fov_x, fov_y);
        spdlog::debug("Camera configured with width={}, height={}, fov_x={}, fov_y={}", width, height, fov_x, fov_y);
    }


    // Render the object using the given camera pose and save the image.
    void Perception::render(const Eigen::Matrix4f &camera_pose, const std::string &image_save_path) const {
        viewer_->setSize(camera_->getConfig().width, camera_->getConfig().height);
        // Set viewer size to camera dimensions

        // Extract the camera position from the pose matrix
        Eigen::Vector3f camera_position = camera_pose.block<3, 1>(0, 3);

        // Calculate the look-at point
        Eigen::Vector3f look_at_point = camera_position + camera_pose.block<3, 1>(0, 2);

        // Extract the up vector from the pose matrix
        Eigen::Vector3f up_vector = camera_pose.block<3, 1>(0, 1);

        // Set the camera parameters in the PCL visualizer
        viewer_->setCameraPosition(
                camera_position.x(), camera_position.y(), camera_position.z(),
                look_at_point.x(), look_at_point.y(), look_at_point.z(),
                up_vector.x(), up_vector.y(), up_vector.z()
                );

        // Render the scene
        viewer_->spinOnce(100);

        // Capture the screenshot
        viewer_->saveScreenshot(image_save_path);

        spdlog::debug("Rendered image saved at: {}", image_save_path);
    }

    std::shared_ptr<Camera> Perception::getCamera() const {
        return camera_;
    }

    // Loads the mesh from a PLY file and throws an error if the file cannot be loaded.
    void Perception::loadMesh(const std::string &object_path) const {
        if (!pcl::io::loadPolygonFilePLY(object_path, *mesh_)) {
            throw std::runtime_error("Failed to load mesh from: " + object_path);
        }
        spdlog::debug("Loaded mesh from: {}", object_path);
    }

    // Sets initial parameters for the viewer, such as background color and camera parameters.
    void Perception::setupViewer() const {
        viewer_->setBackgroundColor(0, 0, 0); // Set a black background for better visibility.
        viewer_->addPolygonMesh(*mesh_, "mesh"); // Add the loaded mesh to the viewer.
        viewer_->initCameraParameters(); // Initialize default camera parameters.
        viewer_->setSize(camera_->getConfig().width, camera_->getConfig().height); // Set the size of the viewer window.
    }

    // Normalizes the mesh to ensure it fits within a unit cube and is centered at the origin.
    void Perception::normalizeMesh() {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromPCLPointCloud2(mesh_->cloud, *cloud); // Convert the polygon mesh to a point cloud for processing.

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud, centroid); // Calculate the centroid of the cloud.
        centerPointCloud(cloud, centroid); // Translate the cloud so that the centroid is at the origin.

        scalePointCloudToFitUnitCube(cloud); // Scale the cloud so that it fits within a unit cube.
        pcl::toPCLPointCloud2(*cloud, mesh_->cloud); // Convert the processed cloud back to the polygon mesh.

        viewer_->updatePolygonMesh(*mesh_, "mesh"); // Update the mesh in the viewer with the normalized mesh.

        spdlog::debug("Mesh normalization complete.");
    }

    // Centers the point cloud at the origin by adjusting all points relative to the centroid.
    void Perception::centerPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const Eigen::Vector4f &centroid) {
        Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
        translation.block<3, 1>(0, 3) = -centroid.head<3>();
        // Create a translation matrix to move the centroid to the origin.
        pcl::transformPointCloud(*cloud, *cloud, translation); // Apply the translation to the cloud.

        spdlog::debug("Centered point cloud to origin.");
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

        spdlog::debug("Scaled point cloud to fit within a unit cube.");
    }

    /*void Perception::detectAndSetSphere(const cv::Mat &target_image) {
        // auto [center, radius] = processing::vision::SphereDetector::detect(target_image);
        // spdlog::info("Detected object sphere with center=({}, {}) and radius={}", center.x, center.y, radius);
    }*/


}
