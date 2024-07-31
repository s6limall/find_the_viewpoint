// File: sampling/fibonacci.hpp

#ifndef SAMPLING_FIBONACCI_LATTICE_SAMPLER_HPP
#define SAMPLING_FIBONACCI_LATTICE_SAMPLER_HPP

#include <Eigen/Dense>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <vector>
#include "sampling/sampler.hpp"

template<typename T = double>
class FibonacciLatticeSampler final : public Sampler<T> {
public:
    FibonacciLatticeSampler(const std::vector<T> &lower_bounds, const std::vector<T> &upper_bounds) :
        Sampler<T>(lower_bounds, upper_bounds) {}

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    generate(size_t num_samples, typename Sampler<T>::TransformFunction transform = nullptr) override {
        this->samples_.resize(this->dimensions_, num_samples);
        T phi = (1 + std::sqrt(5)) / 2; // golden ratio

        for (size_t i = 0; i < num_samples; ++i) {
            T theta = 2 * M_PI * (i / phi - std::floor(i / phi));
            T z = 1 - 2 * static_cast<T>(i) / (num_samples - 1);
            T r = std::sqrt(1 - z * z);

            Eigen::Matrix<T, Eigen::Dynamic, 1> point(this->dimensions_);
            point(0) = r * std::cos(theta); // x
            point(1) = r * std::sin(theta); // y
            if (this->dimensions_ > 2) {
                point(2) = z; // z
                for (size_t d = 3; d < this->dimensions_; ++d) {
                    point(d) = 0; // zero padding for higher dimensions
                }
            }

            if (transform) {
                point = transform(point);
            }

            this->samples_.col(i) = this->mapToBounds(point);

            // Radius verification: ensure the point is on the unit sphere
            T radius = std::sqrt(point(0) * point(0) + point(1) * point(1) + point(2) * point(2));
            assert(std::abs(radius - 1.0) < 1e-6 && "Point is not on the unit sphere.");
        }

        assert(this->samples_.cols() == num_samples && "Number of generated samples must match the requested number.");
        assert(this->samples_.rows() == this->dimensions_ && "Each sample must have the correct number of dimensions.");


        return this->samples_;
    }

    void visualize() const {
        assert(this->samples_.cols() > 0 && "No samples to visualize. Generate samples first.");

        // Create a blank image
        constexpr int width = 800, height = 800;
        cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
        image.setTo(cv::Scalar(255, 255, 255)); // White background

        // Draw the unit sphere
        cv::circle(image, cv::Point(width / 2, height / 2), width / 2 - 50, cv::Scalar(0, 0, 0), 2);

        // Draw the points
        for (int i = 0; i < this->samples_.cols(); ++i) {
            Eigen::Vector3d point = this->samples_.col(i).template head<3>();
            const cv::Point pt(static_cast<int>((point(0) + 1) * width / 2),
                               static_cast<int>((point(1) + 1) * height / 2));

            cv::circle(image, pt, 3, cv::Scalar(0, 0, 255), cv::FILLED);
        }

        // Show the image
        cv::imshow("Fibonacci Lattice Sampling", image);
        cv::waitKey(0); // Wait for a key press to close the window
    }
    void visualize3d() const {
        assert(this->samples_.cols() > 0 && "No samples to visualize. Generate samples first.");

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        for (int i = 0; i < this->samples_.cols(); ++i) {
            pcl::PointXYZ point;
            point.x = this->samples_(0, i);
            point.y = this->samples_(1, i);
            point.z = this->samples_(2, i);
            cloud->points.push_back(point);
        }

        pcl::visualization::PCLVisualizer::Ptr viewer(
                new pcl::visualization::PCLVisualizer("Fibonacci Lattice Sampling"));
        viewer->setBackgroundColor(0, 0, 0);
        viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
        viewer->addCoordinateSystem(1.0);
        viewer->initCameraParameters();

        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
};

#endif // SAMPLING_FIBONACCI_LATTICE_SAMPLER_HPP
