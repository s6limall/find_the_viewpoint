// core/perception.hpp

#ifndef PERCEPTION_HPP
#define PERCEPTION_HPP

#include <string>
#include <memory>

#include <pcl/PolygonMesh.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "common/logging/logger.hpp"
#include "config/configuration.hpp"

#include "core/camera.hpp"

namespace core {
    class Perception {
    public:
        // Constructor to initialize Perception with a given object path.
        explicit Perception(const std::string &object_path);

        ~Perception();

        // Configure the camera with specified width, height, and field of view.
        void configureCamera(int width, int height, float fov_x, float fov_y) const;

        // Render the object using the given camera pose and save the image.
        void render(const Eigen::Matrix4f &camera_pose, const std::string &image_save_path) const;

        // Provide camera to be used in views
        [[nodiscard]] std::shared_ptr<Camera> getCamera() const;

    private:
        std::shared_ptr<pcl::visualization::PCLVisualizer> viewer_; // Visualizer for rendering the mesh
        pcl::PolygonMesh::Ptr mesh_; // Polygon mesh loaded from a file
        std::shared_ptr<Camera> camera_; // Camera object for rendering

        // Load a polygon mesh from a PLY file.
        void loadMesh(const std::string &object_path) const;

        // Configure the PCL visualizer.
        void setupViewer() const;

        // Normalize the mesh to fit within a unit cube at the origin.
        void normalizeMesh();

        // Center the point cloud at the origin.
        void centerPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const Eigen::Vector4f &centroid);

        // Scale the point cloud to fit within a unit cube.
        void scalePointCloudToFitUnitCube(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
    };
}

#endif // PERCEPTION_HPP
