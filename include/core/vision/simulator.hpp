// File: core/simulator.hpp

#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

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
#include "core/vision/perception.hpp"

#include "common/io/image.hpp"
#include "core/camera.hpp"

namespace core {
    class Simulator final : public Perception {
    public:
        enum class NormalizationMethod { UnitCube, UnitSphere };

        Simulator();
        cv::Mat render(const Eigen::Matrix4d &extrinsics, std::string_view save_path) override;

        [[nodiscard]] std::shared_ptr<pcl::visualization::PCLVisualizer> getViewer();

    private:
        std::shared_ptr<pcl::visualization::PCLVisualizer> viewer_; // Visualizer for rendering the mesh
        std::shared_ptr<pcl::PolygonMesh> mesh_; // Polygon mesh loaded from a file

        void configureCamera();
        void loadMesh(std::string_view object_path);

        void setupViewer();

        // Normalize the mesh to fit within a unit cube at the origin.
        void normalizeMesh(NormalizationMethod method = NormalizationMethod::UnitCube) const;

        // Center the point cloud at the origin.
        static void centerPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const Eigen::Vector4d &centroid);

        // Scale the point cloud to fit within a unit cube/sphere.
        static void scalePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, double scale);

        static double calculateScaleForUnitCube(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

        static double calculateScaleForUnitSphere(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);


        struct Defaults {
            static constexpr double fov_x = 0.95;
            static constexpr double fov_y = 0.75;
            static constexpr int width = 640;
            static constexpr int height = 480;
            static constexpr std::string_view mesh_path = "./3d_models/obj_000020.ply";
        };
    };
} // namespace core

#endif // SIMULATOR_HPP
