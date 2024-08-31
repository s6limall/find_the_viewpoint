// File: core/simulator.hpp

#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <memory>
#include <mutex>
#include <string>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <pcl/PolygonMesh.h>
#include <pcl/common/transforms.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "core/camera.hpp"
#include "core/vision/perception.hpp"

#include "common/io/image.hpp"
#include "common/logging/logger.hpp"
#include "config/configuration.hpp"


namespace core {

    class Simulator final : public Perception {
    public:
        enum class NormalizationMethod { UnitCube, UnitSphere };

        static std::shared_ptr<Simulator> create(std::string_view mesh_path = {});

        Simulator(const Simulator &) = delete;
        Simulator &operator=(const Simulator &) = delete;
        Simulator(Simulator &&) = delete;
        Simulator &operator=(Simulator &&) = delete;
        ~Simulator() override = default;

        void loadMesh(std::string_view object_path);

        [[nodiscard]] cv::Mat render(const Eigen::Matrix4d &extrinsics, std::string_view save_path) override;

        [[nodiscard]] std::shared_ptr<Camera> getCamera() const noexcept override { return camera_; }
        [[nodiscard]] auto getViewer() const noexcept { return viewer_; }

    private:
        explicit Simulator(std::string_view mesh_path);

        void setupViewer();
        void updateViewer() const;
        void normalizeMesh(NormalizationMethod method = NormalizationMethod::UnitCube) const;
        static void centerPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const Eigen::Vector4d &centroid);
        static void scalePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, double scale);
        static double calculateScaleForUnitCube(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
        static double calculateScaleForUnitSphere(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

        std::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;
        std::shared_ptr<pcl::PolygonMesh> mesh_;
        mutable std::mutex mutex_;
    };

} // namespace core

#endif // SIMULATOR_HPP
