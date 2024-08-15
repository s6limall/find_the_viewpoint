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
    static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");

public:
    FibonacciLatticeSampler(const std::vector<T> &lower_bounds, const std::vector<T> &upper_bounds, T radius = 1.0) :
        Sampler<T>(lower_bounds, upper_bounds), radius_(radius) {
        if (this->dimensions_ < 2 || this->dimensions_ > 3) {
            throw std::invalid_argument("FibonacciLatticeSampler supports only 2D or 3D.");
        }
        if (radius_ <= 0) {
            throw std::invalid_argument("Radius must be positive");
        }
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    generate(size_t num_samples, typename Sampler<T>::TransformFunction transform = nullptr) override {
        if (num_samples == 0) {
            throw std::invalid_argument("Number of samples must be positive");
        }

        constexpr T golden_ratio = (1 + std::sqrt(5.0)) / 2;
        constexpr T angle_increment = 2 * M_PI / golden_ratio;

        auto generate_point = [this, angle_increment](size_t i, T inv_samples) {
            T t = static_cast<T>(i) * inv_samples;
            // Only cover the upper hemisphere (0 - pi/2)
            T inclination = std::acos(1 - t); // Changed from (1 - 2 * t) to (1 - t)
            T azimuth = angle_increment * i;
            T sin_inclination = std::sin(inclination);

            Eigen::Vector<T, Eigen::Dynamic> point(this->dimensions_);
            point[0] = radius_ * sin_inclination * std::cos(azimuth);
            point[1] = radius_ * sin_inclination * std::sin(azimuth);
            if (this->dimensions_ == 3) {
                point[2] = radius_ * std::cos(inclination);
            }
            return point;
        };

        this->samples_.resize(this->dimensions_, num_samples);
        T inv_samples = 1.0 / static_cast<double>(num_samples);

        for (size_t i = 0; i < num_samples; ++i) {
            auto point = generate_point(i, inv_samples);
            if (transform) {
                point = transform(point);
            }
            this->samples_.col(i) = this->mapToBounds(point);
        }

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

private:
    T radius_;
};

#endif // SAMPLING_FIBONACCI_LATTICE_SAMPLER_HPP
