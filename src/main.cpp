// main.cpp

#include <iostream>
#include "config/configuration.hpp"
#include "common/logging/logger.hpp"
#include "tasks/task_manager.hpp"
#include "sampling/halton_sampler.hpp"
#include "sampling/lhs_sampler.hpp"
#include <stdexcept>
#include <iomanip>
#include <vector>
#include <numeric>
#include <cmath>

#include "processing/image_processor.hpp"
#include "processing/vision/distance_estimator.hpp"

#include <Eigen/Dense>

#include "common/io/image.hpp"
#include "sampling/constrained_spherical_sampler.hpp"

// Function to print samples
void printSamples(const std::vector<std::vector<double> > &samples) {
    for (const auto &sample: samples) {
        for (double coordinate: sample) {
            std::cout << std::fixed << std::setprecision(5) << coordinate << " ";
        }
        std::cout << std::endl;
    }
}

// Function to compute the discrepancy of the sample set
double computeDiscrepancy(const std::vector<std::vector<double> > &samples, int dimension) {
    double max_discrepancy = 0.0;
    int num_samples = samples.size();

    for (int i = 0; i < num_samples; ++i) {
        for (int j = i + 1; j < num_samples; ++j) {
            double discrepancy = 0.0;
            for (int d = 0; d < dimension; ++d) {
                discrepancy += std::fabs(samples[i][d] - samples[j][d]);
            }
            max_discrepancy = std::max(max_discrepancy, discrepancy);
        }
    }

    return max_discrepancy / num_samples;
}

// Test function for HaltonSampler in 6 dimensions
void testHaltonSampler() {
    sampling::HaltonSampler haltonSampler;

    LOG_DEBUG("Testing HaltonSampler in 6 dimensions.");
    size_t num_samples = 50; // Number of samples to generate
    std::vector<double> lower_bounds = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // Lower bounds for each dimension
    std::vector<double> upper_bounds = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // Upper bounds for each dimension

    // Generate samples with adaptive mode off
    std::vector<std::vector<double> > samples = haltonSampler.generate(num_samples, lower_bounds, upper_bounds);
    std::cout << "Samples without adaptive mode:" << std::endl;
    printSamples(samples);

    // Enable adaptive mode and generate samples
    haltonSampler.setAdaptive(true);
    samples = haltonSampler.generate(num_samples, lower_bounds, upper_bounds);
    std::cout << "\nSamples with adaptive mode:" << std::endl;
    printSamples(samples);

    // Check if samples are within bounds
    bool within_bounds = true;
    for (const auto &sample: samples) {
        for (size_t i = 0; i < sample.size(); ++i) {
            if (sample[i] < lower_bounds[i] || sample[i] > upper_bounds[i]) {
                within_bounds = false;
                break;
            }
        }
    }

    if (within_bounds) {
        std::cout << "\nAll samples are within the specified bounds." << std::endl;
    } else {
        std::cout << "\nSome samples are out of the specified bounds!" << std::endl;
    }
}

// Test function for ConstrainedSphericalSampler
void testConstrainedSphericalSampler() {
    constexpr double radius = 1.0;
    constexpr double tolerance = 0.2;
    constexpr size_t num_samples = 50;
    constexpr size_t dimensions = 3;

    cout << "Testing ConstrainedSphericalSampler with radius: " << radius << " and tolerance: " << tolerance << endl;
    sampling::ConstrainedSphericalSampler sampler(radius, tolerance);

    cout << "Generating samples within spherical shell using Halton sequences." << endl;

    std::vector<std::vector<double> > samples = sampler.generate(num_samples, dimensions);
    std::cout << "Samples within spherical shell:" << std::endl;
    printSamples(samples);

    // Check if samples are within the spherical shell
    bool within_shell = true;
    for (const auto &sample: samples) {
        double distance_squared = 0.0;
        for (const double value: sample) {
            distance_squared += value * value;
        }
        const double distance = std::sqrt(distance_squared);
        if (distance < radius * (1.0 - tolerance) || distance > radius * (1.0 + tolerance)) {
            within_shell = false;
            break;
        }
    }

    if (within_shell) {
        std::cout << "\nAll samples are within the spherical shell." << std::endl;
    } else {
        std::cout << "\nSome samples are out of the spherical shell!" << std::endl;
    }
}

void testDistanceEstimator(const std::string &target_image_path) {
    // cv::Mat target_image = cv::imread("../task1/viewspace_images/obj_000020/rgb_2.png", cv::IMREAD_GRAYSCALE);
    const cv::Mat target_image = common::io::image::readImage(target_image_path);

    if (target_image.empty()) {
        LOG_ERROR("Error: Unable to load target image!");
    }

    constexpr float unit_cube_size = 1.0; // Since the mesh is normalized to a unit cube
    constexpr float focal_length = 60.0;
    processing::vision::DistanceEstimator distance_estimator(focal_length, unit_cube_size);

    try {
        double estimated_distance = distance_estimator.estimate(target_image);
        LOG_INFO("Estimated Distance: {}", estimated_distance);
    } catch (const std::exception &e) {
        LOG_ERROR("Distance estimation failed: {}", e.what());
    }

}

int main() {
    try {

        /*testDistanceEstimator("../task1/viewspace_images/obj_000020/rgb_0.png");
        testDistanceEstimator("../task1/viewspace_images/obj_000020/rgb_1.png");
        testDistanceEstimator("../task1/viewspace_images/obj_000020/rgb_2.png");
        testDistanceEstimator("../task1/viewspace_images/obj_000020/rgb_3.png");*/
        // return 0;

        testDistanceEstimator("../task1/viewspace_images/obj_000020/rgb_0.png");
        testDistanceEstimator("../task1/viewspace_images/obj_000020/rgb_1.png");
        testDistanceEstimator("../task1/viewspace_images/obj_000020/rgb_2.png");
        testDistanceEstimator("../task1/viewspace_images/obj_000020/rgb_3.png");
        testHaltonSampler();
        testConstrainedSphericalSampler();
        // return 0;

        // Load task parameters from configuration
        const auto object = config::get("task.objects", "obj_000020");
        const auto test_num = config::get("task.test_num", 1);

        const auto task_manager = tasks::TaskManager::getInstance();
        LOG_INFO("Executing task for object: {}", object);
        task_manager->execute(object, test_num);

        LOG_INFO("All tasks executed successfully.");
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        LOG_CRITICAL("Critical error: {}", e.what());
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred." << std::endl;
        LOG_CRITICAL("Unknown critical error.");
        return 1;
    }

    return 0;
}
