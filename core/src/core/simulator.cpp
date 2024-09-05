// File: core/vision/simulator.cpp

#include "core/vision/simulator.hpp"

namespace core {

    std::shared_ptr<Simulator> Simulator::create(const std::string_view mesh_path) {
        return std::shared_ptr<Simulator>(new Simulator(mesh_path));
    }

    Simulator::Simulator(const std::string_view mesh_path) {
        const int width = 640;
        const int height = 480;
        const auto fov_x = 0.95;
        const auto fov_y = 0.75;

        camera_ = std::make_shared<Camera>();
        camera_->setIntrinsics(width, height, fov_x, fov_y);
        LOG_INFO("Camera configured: width={}, height={}, fov_x={}, fov_y={}", width, height, fov_x, fov_y);

        setupViewer();

        if (!mesh_path.empty()) {
            loadMesh(mesh_path);
        }
    }

    void Simulator::setupViewer() {
        viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>("Simulator Viewer");
        viewer_->setBackgroundColor(255, 255, 255);
        viewer_->initCameraParameters();
        viewer_->setSize(camera_->getIntrinsics().width, camera_->getIntrinsics().height);
    }

    void Simulator::loadMesh(std::string_view object_path) {
        std::lock_guard<std::mutex> lock(mutex_);

        mesh_ = std::make_shared<pcl::PolygonMesh>();
        if (pcl::io::loadPolygonFilePLY(object_path.data(), *mesh_)) {
            LOG_INFO("Mesh loaded from: {}", object_path);
            updateViewer();
            normalizeMesh();
        } else {
            mesh_.reset();
            throw std::runtime_error(fmt::format("Failed to load mesh from: {}", object_path));
        }
    }

    void Simulator::updateViewer() const {
        viewer_->removeAllPointClouds();
        viewer_->removeAllShapes();
        if (mesh_) {
            viewer_->addPolygonMesh(*mesh_, "mesh");
        }
    }

    void Simulator::normalizeMesh(const NormalizationMethod method) const {
        if (!mesh_) {
            LOG_WARN("No mesh loaded. Skipping normalization.");
            return;
        }

        const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2(mesh_->cloud, *cloud);

        if (cloud->empty()) {
            LOG_WARN("Point cloud is empty. Skipping normalization.");
            return;
        }

        Eigen::Vector4d centroid;
        pcl::compute3DCentroid(*cloud, centroid);
        centerPointCloud(cloud, centroid);

        const double scale = (method == NormalizationMethod::UnitCube) ? calculateScaleForUnitCube(cloud)
                                                                       : calculateScaleForUnitSphere(cloud);
        if (scale == 0.0) {
            LOG_WARN("Scale is zero. Skipping scaling.");
            return;
        }

        scalePointCloud(cloud, scale);

        pcl::toPCLPointCloud2(*cloud, mesh_->cloud);
        viewer_->updatePolygonMesh(*mesh_, "mesh");
        LOG_DEBUG("Mesh normalization complete.");
    }

    void Simulator::centerPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                     const Eigen::Vector4d &centroid) {
        Eigen::Matrix4d translation = Eigen::Matrix4d::Identity();
        translation.block<3, 1>(0, 3) = -centroid.head<3>();
        pcl::transformPointCloud(*cloud, *cloud, translation);
        LOG_DEBUG("Centered point cloud to origin.");
    }

    void Simulator::scalePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, double scale) {
        Eigen::Matrix4d scale_matrix = Eigen::Matrix4d::Identity();
        scale_matrix.block<3, 3>(0, 0) *= scale;
        pcl::transformPointCloud(*cloud, *cloud, scale_matrix);
        LOG_DEBUG("Scaled point cloud with scale factor: {}", scale);
    }

    double Simulator::calculateScaleForUnitCube(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        Eigen::Vector4f min_point, max_point;
        pcl::getMinMax3D(*cloud, min_point, max_point);

        const double max_dim =
                std::max({max_point.x() - min_point.x(), max_point.y() - min_point.y(), max_point.z() - min_point.z()});
        return max_dim == 0.0 ? 0.0 : 1.0 / max_dim;
    }

    double Simulator::calculateScaleForUnitSphere(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        double max_distance = 0.0;
        for (const auto &point: cloud->points) {
            max_distance = std::max(max_distance, static_cast<double>(point.getVector3fMap().norm()));
        }
        return max_distance == 0.0 ? 0.0 : 1.0 / max_distance;
    }

    cv::Mat Simulator::render(const Eigen::Matrix4d &extrinsics, std::string_view save_path) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!mesh_) {
            throw std::runtime_error("No mesh loaded. Cannot render.");
        }

        const auto duration = 0;

        viewer_->setSize(camera_->getIntrinsics().width, camera_->getIntrinsics().height);
        viewer_->setCameraParameters(camera_->getIntrinsics().getMatrix().cast<float>(), extrinsics.cast<float>());
        viewer_->spinOnce(duration);
        viewer_->saveScreenshot(std::string(save_path));

        cv::Mat rendered_image = common::io::image::readImage(save_path);
        if (rendered_image.empty()) {
            throw std::runtime_error(fmt::format("Failed to read rendered image from: {}", save_path));
        }

        LOG_TRACE("Image rendered and saved at: {}", save_path);
        return rendered_image;
    }

} // namespace core
