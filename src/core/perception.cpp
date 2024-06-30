// File: core/perception.cpp

#include "core/perception.hpp"

namespace core {
    Perception::Perception(const std::string &object_path) :
        viewer_(std::make_shared<pcl::visualization::PCLVisualizer>("Viewer")),
        mesh_(std::make_shared<pcl::PolygonMesh>()),
        camera_(std::make_shared<Camera>()) {

        auto width = config::get<int>("camera.width", 640);
        auto height = config::get<int>("camera.height", 480);
        auto fov_x = config::get<float>("camera.fov_x", 0.95f);
        auto fov_y = config::get<float>("camera.fov_y", 0.75f);

        LOG_INFO("Retrieved camera settings from configuration: width={}, height={}, fov_x={}, fov_y={}", width, height,
                 fov_x, fov_y);

        // Configure the camera with these settings
        configureCamera(width, height, fov_x, fov_y);

        loadMesh(object_path); // Load the 3D model mesh from a file.
        setupViewer(); // Setup the PCL visualizer.
        normalizeMesh(); // Normalize the mesh to a unit cube centered at the origin.

        LOG_DEBUG("Perception initialized with camera intrinsics: width={}, height={}, fov_x={}, fov_y={}", width,
                  height, fov_x, fov_y);
    }

    Perception::~Perception() {
        if (!viewer_ || viewer_->wasStopped()) {
            viewer_->close(); // Close the PCL visualizer.
        }
        std::cout << "Perception object destroyed." << endl;
    }

    // Configures camera parameters and updates the intrinsic matrix.
    void Perception::configureCamera(int width, int height, float fov_x, float fov_y) const {
        LOG_DEBUG("Configuring camera with width={}, height={}, fov_x={}, fov_y={}", width, height, fov_x, fov_y);
        camera_->setIntrinsics(width, height, fov_x, fov_y);
    }


    // Render the object using the given camera pose and save the image.
    void Perception::render(const Eigen::Matrix4f &camera_pose, const std::string &image_save_path) const {
        LOG_DEBUG("Rendering image...");

        // Set viewer size to camera dimensions
        viewer_->setSize(camera_->getParameters().width, camera_->getParameters().height);

        // Extract the camera position from the pose matrix (translation - x, y, z)
        Eigen::Vector3f camera_position = camera_pose.block<3, 1>(0, 3); // camera_->getPosition();
        LOG_DEBUG("Camera position: ({}, {}, {})", camera_position.x(), camera_position.y(), camera_position.z());

        // Calculate the look-at point (object center)
        Eigen::Vector3f look_at_point = camera_position + camera_pose.block<3, 1>(0, 2);
        LOG_DEBUG("Camera looking at point: ({}, {}, {})", look_at_point.x(), look_at_point.y(), look_at_point.z());

        // Extract the up vector from the pose matrix
        Eigen::Vector3f up_vector = camera_pose.block<3, 1>(0, 1);

        viewer_->setCameraParameters()

        // Set the camera parameters in the PCL visualizer
        viewer_->setCameraPosition(
                camera_position.x(), camera_position.y(), camera_position.z(),
                look_at_point.x(), look_at_point.y(), look_at_point.z(),
                up_vector.x(), up_vector.y(), up_vector.z()
                );

        viewer_->spinOnce(100); // Render for 100ms
        viewer_->saveScreenshot(image_save_path);

        LOG_DEBUG("Rendered image saved at: {}", image_save_path);
    }

    std::shared_ptr<Camera> Perception::getCamera() const {
        return camera_;
    }

    // Loads the mesh from a PLY file and throws an error if the file cannot be loaded.
    void Perception::loadMesh(const std::string &object_path) const {
        if (!pcl::io::loadPolygonFilePLY(object_path, *mesh_)) {
            throw std::runtime_error("Failed to load mesh from: " + object_path);
        }
        LOG_DEBUG("Loaded mesh from: {}", object_path);
    }

    // Sets initial parameters for the viewer, such as background color and camera parameters.
    void Perception::setupViewer() const {
        viewer_->setBackgroundColor(255, 255, 255); // (0,0,0) for black, (255,255,255) for white.
        viewer_->addPolygonMesh(*mesh_, "mesh"); // Add the loaded mesh to the viewer.
        viewer_->initCameraParameters(); // Initialize default camera parameters.
        viewer_->setSize(camera_->getParameters().width, camera_->getParameters().height); // Window size.
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

        LOG_DEBUG("Mesh normalization complete.");
    }

    // Centers the point cloud at the origin by adjusting all points relative to the centroid.
    void Perception::centerPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const Eigen::Vector4f &centroid) {
        Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
        translation.block<3, 1>(0, 3) = -centroid.head<3>();
        // Create a translation matrix to move the centroid to the origin.
        pcl::transformPointCloud(*cloud, *cloud, translation); // Apply the translation to the cloud.

        LOG_DEBUG("Centered point cloud to origin.");
    }

    // Scales the point cloud to fit within a unit cube by determining the maximum dimension and scaling all dimensions equally.
    void Perception::scalePointCloudToFitUnitCube(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        Eigen::Vector4f min_point, max_point;
        pcl::getMinMax3D(*cloud, min_point, max_point); // Determine the min and max points in the cloud.

        const float max_dim = std::max(max_point.x() - min_point.x(),
                                       std::max(max_point.y() - min_point.y(), max_point.z() - min_point.z()));
        Eigen::Matrix4f scale = Eigen::Matrix4f::Identity();
        scale.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity() / max_dim;

        // Calculate a scale matrix to fit the cloud within a unit cube.
        pcl::transformPointCloud(*cloud, *cloud, scale); // Apply the scale transformation to the cloud.

        LOG_DEBUG("Scaled point cloud to fit within a unit cube.");
    }

}
