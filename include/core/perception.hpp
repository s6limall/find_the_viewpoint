// core/perception.hpp

#ifndef PERCEPTION_HPP
#define PERCEPTION_HPP

#include <future>
#include <memory>
#include <string>

#include <pcl/PolygonMesh.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "common/logging/logger.hpp"
#include "config/configuration.hpp"

#include "common/io/image.hpp"
#include "core/camera.hpp"

namespace core {
    class Perception {
    public:
        enum class NormalizationMethod { UnitCube, UnitSphere };

        // Constructor to initialize Perception with a given object path.
        // explicit Perception(const std::string &object_path);

        Perception(const Perception &) = delete;

        Perception &operator=(const Perception &) = delete;

        // Render the object using the given camera pose and save the image.
        static cv::Mat render(const Eigen::Matrix4d &extrinsics, std::string_view image_save_path = "render_tmp.jpg");

        // Overloaded render function to return the rendered image as cv::Mat
        // [[nodiscard]] static cv::Mat render(const Eigen::Matrix4d &extrinsics);

        // Provide camera to be used in views
        [[nodiscard]] static std::shared_ptr<Camera> getCamera();

        // Provide access to the viewer
        [[nodiscard]] static std::shared_ptr<pcl::visualization::PCLVisualizer> getViewer();

    private:
        static std::shared_ptr<pcl::visualization::PCLVisualizer> viewer_; // Visualizer for rendering the mesh
        static std::shared_ptr<pcl::PolygonMesh> mesh_; // Polygon mesh loaded from a file
        static std::shared_ptr<Camera> camera_; // Camera object for rendering
        static std::once_flag init_flag_; // Flag to ensure static initialization is done only once

        Perception() = default;

        ~Perception() = default;

        static void initialize();

        // Configure the camera with specified width, height, and field of view.
        static void configureCamera();

        // Load a polygon mesh from a PLY file.
        static void loadMesh(std::string_view object_path);

        // Configure the PCL visualizer.
        static void setupViewer();

        // Normalize the mesh to fit within a unit cube at the origin.
        static void normalizeMesh(NormalizationMethod method = NormalizationMethod::UnitCube);

        // Center the point cloud at the origin.
        static void centerPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const Eigen::Vector4d &centroid);

        // Scale the point cloud to fit within a unit cube/sphere.
        static void scalePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, double scale);

        // Scale the point cloud to fit within a unit cube.
        static double calculateScaleForUnitCube(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

        // Scale the point cloud to fit within a unit sphere.
        static double calculateScaleForUnitSphere(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);


        struct Defaults {
            static constexpr double fov_x = 0.95;
            static constexpr double fov_y = 0.75;
            static constexpr int width = 640;
            static constexpr int height = 480;
            static constexpr std::string_view mesh_path = "./3d_models/obj_000020.ply";

            static std::string_view getMeshPath() noexcept { return mesh_path; }

            static int getWidth() { return width; }

            static int getHeight() { return height; }

            static int getFoVx() { return fov_x; }

            static int getFoVy() { return fov_y; }

            static std::string_view getImageSavePath() { return "render_tmp.jpg"; }

            static std::string_view getViewerName() { return "Viewer"; }
        };
    };
} // namespace core

#endif // PERCEPTION_HPP
