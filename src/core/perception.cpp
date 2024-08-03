// File: core/perception.cpp

#include "core/perception.hpp"

namespace core {

    std::shared_ptr<pcl::visualization::PCLVisualizer> Perception::viewer_ = nullptr;
    std::shared_ptr<pcl::PolygonMesh> Perception::mesh_ = nullptr;
    std::shared_ptr<Camera> Perception::camera_ = nullptr;
    std::once_flag Perception::init_flag_;

    void Perception::initialize() {
        configureCamera();
        loadMesh(config::get("paths.mesh", Defaults::mesh_path.data()));
        setupViewer();
        normalizeMesh();
    }

    void Perception::configureCamera() {
        camera_ = std::make_shared<Camera>();
        const int width = config::get("camera.width", Defaults::width);
        const int height = config::get("camera.height", Defaults::height);
        const auto fov_x = config::get("camera.fov_x", Defaults::fov_x);
        const auto fov_y = config::get("camera.fov_y", Defaults::fov_y);

        LOG_INFO("Configuring camera with width={}, height={}, fov_x={}, fov_y={}", width, height, fov_x, fov_y);
        camera_->setIntrinsics(width, height, fov_x, fov_y);
    }

    void Perception::loadMesh(std::string_view object_path) {
        mesh_ = std::make_shared<pcl::PolygonMesh>();
        if (!pcl::io::loadPolygonFilePLY(object_path.data(), *mesh_)) {
            throw std::runtime_error("Failed to load mesh from: " + std::string(object_path));
        }
        LOG_DEBUG("Loaded mesh from: {}", object_path);
    }

    void Perception::setupViewer() {
        viewer_ = std::make_shared<pcl::visualization::PCLVisualizer>("Viewer");
        viewer_->setBackgroundColor(255, 255, 255);
        viewer_->addPolygonMesh(*mesh_, "mesh");
        viewer_->initCameraParameters();
        viewer_->setSize(camera_->getIntrinsics().width, camera_->getIntrinsics().height);
    }

    void Perception::normalizeMesh(NormalizationMethod method) {
        const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
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

    void Perception::centerPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                      const Eigen::Vector4d &centroid) {
        Eigen::Matrix4d translation = Eigen::Matrix4d::Identity();
        translation.block<3, 1>(0, 3) = -centroid.head<3>();
        pcl::transformPointCloud(*cloud, *cloud, translation);
        LOG_DEBUG("Centered point cloud to origin.");
    }

    void Perception::scalePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, double scale) {
        Eigen::Matrix4d scale_matrix = Eigen::Matrix4d::Identity();
        scale_matrix.block<3, 3>(0, 0) *= scale;
        pcl::transformPointCloud(*cloud, *cloud, scale_matrix);
        LOG_DEBUG("Scaled point cloud with scale factor: {}", scale);
    }

    double Perception::calculateScaleForUnitCube(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        Eigen::Vector4f min_point, max_point;
        pcl::getMinMax3D(*cloud, min_point, max_point);

        const double max_dim =
                std::max({max_point.x() - min_point.x(), max_point.y() - min_point.y(), max_point.z() - min_point.z()});
        return max_dim == 0.0 ? 0.0 : 1.0 / max_dim;
    }

    double Perception::calculateScaleForUnitSphere(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        double max_distance = 0.0;
        for (const auto &point: cloud->points) {
            max_distance = std::max(max_distance, static_cast<double>(point.getVector3fMap().norm()));
        }
        return max_distance == 0.0 ? 0.0 : 1.0 / max_distance;
    }

    cv::Mat Perception::render(const Eigen::Matrix4d &extrinsics, std::string_view image_save_path) {
        std::call_once(init_flag_, &Perception::initialize);

        const auto duration = config::get("rendering.duration", 10);

        LOG_TRACE("Rendering image...");

        try {
            viewer_->setSize(camera_->getIntrinsics().width, camera_->getIntrinsics().height);
            viewer_->setCameraParameters(camera_->getIntrinsics().getMatrix().cast<float>(), extrinsics.cast<float>());
            viewer_->spinOnce(duration);
            viewer_->saveScreenshot(std::string(image_save_path));

            LOG_TRACE("Rendered image saved at: {}", image_save_path);

            cv::Mat rendered_image;
            try {
                rendered_image = common::io::image::readImage(image_save_path);
                if (rendered_image.empty()) {
                    LOG_ERROR("Rendered image is empty.");
                    throw std::runtime_error("Rendered image is empty.");
                }
            } catch (const std::exception &e) {
                LOG_ERROR("Failed to read the rendered image from {}: {}", image_save_path, e.what());
                throw;
            }

            LOG_TRACE("Rendered image captured.");
            return rendered_image;
        } catch (const std::exception &e) {
            LOG_ERROR("Rendering task failed: {}", e.what());
            return {}; // Return an empty image in case of failure
        }
    }

    std::shared_ptr<Camera> Perception::getCamera() {
        std::call_once(init_flag_, &Perception::initialize);
        return camera_;
    }

    std::shared_ptr<pcl::visualization::PCLVisualizer> Perception::getViewer() {
        std::call_once(init_flag_, &Perception::initialize);
        return viewer_;
    }

} // namespace core
