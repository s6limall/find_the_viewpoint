// File: core/renderer.cpp

#include "core/renderer.hpp"

#include "config/configuration.hpp"


namespace core {

    Renderer::Renderer() :
        viewer_(std::make_shared<pcl::visualization::PCLVisualizer>("Viewer")),
        mesh_(std::make_shared<pcl::PolygonMesh>()),
        initialized_(false),
        rendered_(false),
        configured_(false) {
        auto object_name = config::get("task.objects", "obj_000020");
        auto model_directory = config::get("paths.model_directory", "../3d_models");
        std::string model_path = fmt::format("{}/{}.ply", model_directory, object_name);
        instance().loadMesh(model_path);

        LOG_INFO("Renderer created with default parameters.");
    }

    Renderer::~Renderer() {
        if (viewer_ && !viewer_->wasStopped()) {
            viewer_->close();
        }
        LOG_INFO("Renderer destroyed.");
    }

    Renderer &Renderer::configure(int width, int height, double fov_x, double fov_y) {
        Renderer &renderer = instance();
        // renderer.reset();
        renderer.camera_.setIntrinsics(width, height, fov_x, fov_y);

        renderer.configured_ = true;
        LOG_INFO("Renderer configured with parameters: width={}, height={}, fov_x={}, fov_y={}", width, height, fov_x,
                 fov_y);
        return renderer;
    }

    cv::Mat Renderer::render(const Eigen::Matrix4d &extrinsics, std::string_view image_save_path) {
        Renderer &renderer = instance();

        if (!renderer.mesh_ || renderer.mesh_->cloud.data.empty()) {
            LOG_WARN("Mesh not loaded or is empty. Loading default mesh.");
            throw std::runtime_error("Mesh not loaded or is empty.");

        }

        if (!renderer.configured_) {
            configure();
        }
        renderer.ensureInitialized();


        renderer.viewer_->setCameraParameters(renderer.camera_.getIntrinsics().getMatrix().cast<float>(),
                                              extrinsics.cast<float>());
        renderer.viewer_->spinOnce(100);

        try {
            // Save to file
            renderer.viewer_->saveScreenshot(std::string(image_save_path));
            LOG_INFO("Rendered image saved at: {}", image_save_path);

            // Read image from file and return as cv::Mat
            cv::Mat image = common::io::image::readImage(image_save_path, cv::IMREAD_COLOR);
            if (image.empty()) {
                LOG_ERROR("Failed to read the saved image file: {}", image_save_path);
                throw std::runtime_error("Failed to read the saved image file: " + std::string(image_save_path));
            }
            return image;
        } catch (const std::exception &e) {
            LOG_ERROR("Error during rendering: {}", e.what());
            throw;
        }
    }

    Renderer &Renderer::loadMesh(std::string_view mesh_path, const bool normalize) {
        ensureInitialized();
        if (!pcl::io::loadPolygonFilePLY(std::string(mesh_path), *mesh_)) {
            LOG_ERROR("Failed to load mesh from: {}", mesh_path);
            throw std::runtime_error("Failed to load mesh from: " + std::string(mesh_path));
        }
        LOG_INFO("Mesh loaded from: {}", mesh_path);
        if (normalize) {
            normalizeMesh();
        }
        return *this;
    }

    Renderer &Renderer::loadMesh(const pcl::PolygonMesh &mesh, const bool normalize) {
        ensureInitialized();
        *mesh_ = mesh;
        LOG_INFO("Mesh loaded from memory.");
        if (normalize) {
            normalizeMesh();
        }
        return *this;
    }

    Renderer &Renderer::normalizeMesh(std::string_view method) {
        ensureInitialized();
        if (!mesh_) {
            LOG_ERROR("Mesh not loaded.");
            throw std::runtime_error("Mesh not loaded.");
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromPCLPointCloud2(mesh_->cloud, *cloud);

        Eigen::Vector4d centroid;
        pcl::compute3DCentroid(*cloud, centroid);
        centerPointCloud(cloud, centroid);

        if (method == "UnitCube") {
            scalePointCloudToFitUnitCube(cloud);
        } else if (method == "UnitSphere") {
            scalePointCloudToFitUnitSphere(cloud);
        }

        pcl::toPCLPointCloud2(*cloud, mesh_->cloud);
        LOG_INFO("Mesh normalized using method: {}", method);
        return *this;
    }

    void Renderer::setupViewer() const {

        const auto &intrinsics = camera_.getIntrinsics();
        viewer_->setSize(intrinsics.width, intrinsics.height);
        viewer_->initCameraParameters();
        if (mesh_) {
            viewer_->setBackgroundColor(255, 255, 255);
            viewer_->addPolygonMesh(*mesh_, "mesh");
        }
        LOG_INFO("Viewer setup complete.");
    }

    void Renderer::centerPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                    const Eigen::Vector4d &centroid) {
        Eigen::Matrix4d translation = Eigen::Matrix4d::Identity();
        translation.block<3, 1>(0, 3) = -centroid.head<3>();
        pcl::transformPointCloud(*cloud, *cloud, translation);
        LOG_DEBUG("Point cloud centered to origin.");
    }

    void Renderer::scalePointCloudToFitUnitCube(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        Eigen::Vector4f min_point, max_point;
        pcl::getMinMax3D(*cloud, min_point, max_point);

        const double max_dim = std::max({max_point.x() - min_point.x(),
                                         max_point.y() - min_point.y(),
                                         max_point.z() - min_point.z()});

        Eigen::Matrix4d scale_matrix = Eigen::Matrix4d::Identity();
        scale_matrix.block<3, 3>(0, 0) /= max_dim;
        pcl::transformPointCloud(*cloud, *cloud, scale_matrix);
        LOG_DEBUG("Point cloud scaled to fit within a unit cube.");
    }

    void Renderer::scalePointCloudToFitUnitSphere(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        const double max_distance = std::accumulate(cloud->points.begin(), cloud->points.end(), 0.0,
                                                    [](double max_dist, const pcl::PointXYZ &point) {
                                                        return std::max(static_cast<float>(max_dist),
                                                                        point.getVector3fMap().norm());
                                                    });

        const double scale = 1.0 / max_distance;
        Eigen::Matrix4d scale_matrix = Eigen::Matrix4d::Identity();
        scale_matrix.block<3, 3>(0, 0) *= scale;
        pcl::transformPointCloud(*cloud, *cloud, scale_matrix);
        LOG_DEBUG("Point cloud scaled to fit within a unit sphere.");
    }

    void Renderer::reset() {
        viewer_->removeAllPointClouds();
        viewer_->removeAllShapes();
        viewer_->removeAllCoordinateSystems();
        rendered_ = false;
        configured_ = false;
        LOG_INFO("Renderer reset.");
    }

    void Renderer::ensureInitialized() {
        std::call_once(init_flag_, [this]() {
            setupViewer();
            initialized_ = true;
        });
    }

}
