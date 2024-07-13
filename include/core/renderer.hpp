// File: core/renderer.hpp

#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <string>
#include <string_view>
#include <memory>
#include <mutex>
#include <optional>

#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/PolygonMesh.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "core/camera.hpp"
#include "common/logging/logger.hpp"
#include "common/io/image.hpp"

namespace core {

    class Renderer {
    public:
        Renderer(const Renderer &) = delete;

        Renderer &operator=(const Renderer &) = delete;

        // Static method to initialize/configure the renderer
        static Renderer &configure(int width = 640, int height = 480, double fov_x = 0.95, double fov_y = 0.75);

        // Static method to render with extrinsics, save image, and return cv::Mat
        static cv::Mat render(const Eigen::Matrix4d &extrinsics, std::string_view image_save_path);

        // Load mesh from file path or polygon mesh
        Renderer &loadMesh(std::string_view mesh_path, bool normalize = true);

        Renderer &loadMesh(const pcl::PolygonMesh &mesh, bool normalize = true);

        // Normalize mesh using specified method
        Renderer &normalizeMesh(std::string_view method = "UnitCube");

    private:
        std::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;
        std::shared_ptr<pcl::PolygonMesh> mesh_;
        Camera camera_;
        bool initialized_;
        bool rendered_;
        bool configured_;
        std::once_flag init_flag_;

        Renderer();

        ~Renderer();

        static Renderer &instance();

        static void centerPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const Eigen::Vector4d &centroid);

        static void scalePointCloudToFitUnitCube(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

        static void scalePointCloudToFitUnitSphere(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

        void setupViewer() const;

        void reset();

        void ensureInitialized();
    };

    inline Renderer &Renderer::instance() {
        static Renderer instance;
        return instance;
    }

}
#endif // RENDERER_HPP
