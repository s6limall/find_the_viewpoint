// File: common/utilities/visualizer.cpp


#include "common/utilities/visualizer.hpp"

namespace common::utilities {

    void Visualizer::visualizeResults(const std::vector<ViewPoint<double> > &samples, const double inner_radius,
                                      const double outer_radius) {
        std::vector<Eigen::Matrix4d> poses;
        poses.reserve(samples.size());
        for (const auto &sample: samples) {
            core::View view = sample.toView();
            poses.push_back(view.getPose());
        }

        visualizeViewpoints(poses, inner_radius, outer_radius);
    }


    void Visualizer::visualizeClusters(const std::vector<ViewPoint<double> > &samples) {
        pcl::visualization::PCLVisualizer viewer("Cluster Visualization");

        std::map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;

        // Group samples by cluster id
        for (const auto &sample: samples) {
            auto &cloud = clusters[sample.getClusterId()];
            if (!cloud) {
                cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ> >();
            }
            cloud->emplace_back(sample.getPosition().x(), sample.getPosition().y(), sample.getPosition().z());
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        // Visualize each cluster
        int cluster_id = 0;
        for (const auto &[id, cloud]: clusters) {
            const double r = dis(gen), g = dis(gen), b = dis(gen);
            std::string cloud_name = "cluster_" + std::to_string(cluster_id++);

            viewer.addPointCloud(cloud, cloud_name);
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, cloud_name);
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5.0, cloud_name);
        }

        viewer.spin();
    }


    void Visualizer::visualizeViewpoints(const std::vector<Eigen::Matrix4d> &poses, const double inner_radius,
                                         const double outer_radius) {
        pcl::visualization::PCLVisualizer viewer("Viewpoints Visualization");

        // Add spheres for each pose
        for (const auto &pose: poses) {
            pcl::PointXYZ point(pose(0, 3), pose(1, 3), pose(2, 3));
            viewer.addSphere(point, 0.01, 1.0, 0.0, 0.0, "sphere_" + std::to_string(&pose - &poses[0]));
            // Red spheres for viewpoints
        }

        // Add inner and outer spherical shells
        viewer.addSphere(pcl::PointXYZ(0.0, 0.0, 0.0), inner_radius, "inner_sphere");
        viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                           pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "inner_sphere");
        viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "inner_sphere");
        // Green for inner radius

        viewer.addSphere(pcl::PointXYZ(0.0, 0.0, 0.0), outer_radius, "outer_sphere");
        viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                           pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "outer_sphere");
        viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 1.0, "outer_sphere");
        // Blue for outer radius

        viewer.addCoordinateSystem(1.0);
        viewer.setBackgroundColor(255, 255, 255);
        while (!viewer.wasStopped()) {
            viewer.spinOnce(100);
        }
    }

}
