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

#include "optimization/cma_es_optimizer.hpp"
#include "processing/image_processor.hpp"
#include "processing/vision/distance_estimator.hpp"
#include "viewpoint/provider.hpp"

#include <Eigen/Dense>

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

    int num_samples = 10; // Number of samples to generate
    std::vector<double> lower_bounds = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // Lower bounds for each dimension
    std::vector<double> upper_bounds = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // Upper bounds for each dimension

    // Generate samples with adaptive mode off
    std::vector<std::vector<double> > samples = haltonSampler.generate(num_samples, lower_bounds, upper_bounds, false);
    std::cout << "Samples without adaptive mode:" << std::endl;
    printSamples(samples);

    // Calculate discrepancy without adaptive mode
    double discrepancy = computeDiscrepancy(samples, lower_bounds.size());
    std::cout << "Discrepancy without adaptive mode: " << discrepancy << std::endl;

    // Generate samples with adaptive mode on
    samples = haltonSampler.generate(num_samples, lower_bounds, upper_bounds, true);
    std::cout << "\nSamples with adaptive mode:" << std::endl;
    printSamples(samples);

    // Calculate discrepancy with adaptive mode
    discrepancy = computeDiscrepancy(samples, lower_bounds.size());
    std::cout << "Discrepancy with adaptive mode: " << discrepancy << std::endl;

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

// Test function: Sphere function f(x) = sum(x_i^2)
double sphere_function(const Eigen::VectorXd &x) {
    return x.squaredNorm();
}

// Function to convert Eigen::VectorXd to core::View
core::View createViewFromVector(const Eigen::VectorXd &vec) {
    core::View view;
    Eigen::Vector3f position = vec.head<3>().cast<float>();
    view.computePoseFromPositionAndObjectCenter(position, Eigen::Vector3f(0, 0, 0));
    return view;
}

void test_cma_es() {
    const int dimensions = 3;
    const int population_size = 50;
    const int max_iterations = 100;
    const double sigma = 0.5;
    const double tolerance = 1e-6;

    optimization::CMAESOptimizer optimizer(dimensions, population_size, max_iterations, sigma, tolerance);

    // Create initial views (random points in the space)
    std::vector<core::View> initial_views;
    for (int i = 0; i < population_size; ++i) {
        Eigen::VectorXd random_vec = Eigen::VectorXd::Random(dimensions);
        initial_views.push_back(createViewFromVector(random_vec));
    }

    // Define the evaluation function for the optimizer
    auto evaluate_callback = [&](const core::View &view, const cv::Mat &target_image) -> double {
        Eigen::VectorXd vec = view.toVector();
        return sphere_function(vec);
    };

    // Run optimization
    optimization::OptimizationResult result = optimizer.optimize(initial_views, cv::Mat(), evaluate_callback);

    std::cout << "Best score after optimization: " << result.best_score << std::endl;
    std::cout << "Best view: " << result.optimized_views[0].toVector().transpose() << std::endl;

    // Check if the optimizer converged to the correct minimum
    if (std::abs(result.best_score) < tolerance) {
        std::cout << "CMA-ES optimizer test passed!" << std::endl;
    } else {
        std::cout << "CMA-ES optimizer test failed!" << std::endl;
    }
}


double evaluateView(const core::View &view, const cv::Mat &targetImage,
                    std::shared_ptr<core::Perception> perceptionSimulator) {
    std::string tempImagePath = "../tmp/rendered_view.png";
    perceptionSimulator->render(view.getPose(), tempImagePath);
    cv::Mat renderedImage = cv::imread(tempImagePath, cv::IMREAD_COLOR); // Ensure correct type

    if (renderedImage.empty()) {
        std::cerr << "Error: Rendered image is empty for view." << std::endl;
        return std::numeric_limits<double>::max();
    }

    // Convert renderedImage to match targetImage type if necessary
    if (renderedImage.type() != targetImage.type()) {
        renderedImage.convertTo(renderedImage, targetImage.type());
    }

    // Compare rendered image with target image
    cv::Mat diff;
    cv::absdiff(renderedImage, targetImage, diff);
    cv::Scalar diffSum = cv::sum(diff);
    double score = diffSum[0]; // Example: Sum of absolute differences

    return score;
}


void cma_es() {
    // Example parameters
    std::string objectName = "obj_000020"; // Replace with your object name
    std::string modelDirectory = "../3d_models/";
    std::string viewSpaceFile = "../view_space/5.txt";
    std::string viewSpaceImagesDirectory = "../task1/viewspace_images/" + objectName + "/";
    cv::Mat targetImage = cv::imread(viewSpaceImagesDirectory + "rgb_0.png");

    // Initialize perception simulator
    std::shared_ptr<core::Perception> perceptionSimulator = std::make_shared<core::Perception>(
            modelDirectory + objectName + ".ply");

    // Load or generate viewpoints
    auto viewpointProvider = viewpoint::Provider::createProvider(true, viewSpaceFile, 100, 3);
    std::vector<core::View> viewSpace = viewpointProvider->provision();

    // Initialize CMAESOptimizer
    optimization::CMAESOptimizer optimizer;
    optimizer.initialize();

    // Perform optimization
    optimization::OptimizationResult result = optimizer.optimize(
            viewSpace,
            targetImage,
            [&](const core::View &view, const cv::Mat &targetImage) {
                return evaluateView(view, targetImage, perceptionSimulator);
            }
            );

    // Print results
    std::cout << "Optimization completed." << std::endl;
    std::cout << "Best score: " << result.best_score << std::endl;
    std::cout << "Optimized views:" << std::endl;
    for (const auto &view: result.optimized_views) {
        std::cout << view.getPose().transpose() << std::endl;
    }
}

void testDistanceEstimator(const std::string &target_image_path) {
    // cv::Mat target_image = cv::imread("../task1/viewspace_images/obj_000020/rgb_2.png", cv::IMREAD_GRAYSCALE);
    cv::Mat target_image = cv::imread(target_image_path, cv::IMREAD_GRAYSCALE);
    if (target_image.empty()) {
        spdlog::error("Error: Unable to load target image!");
    }

    double unit_cube_size = 1.0; // Since the mesh is normalized to a unit cube

    processing::vision::DistanceEstimator distance_estimator(unit_cube_size);

    try {
        double estimated_distance = distance_estimator.estimate(target_image);
        spdlog::info("Estimated Distance: {}", estimated_distance);
    } catch (const std::exception &e) {
        spdlog::error("Distance estimation failed: {}", e.what());
    }

}

struct CustomType {
    int a = 10;
    double b = 20.5;
};

std::ostream &operator<<(std::ostream &os, const CustomType &obj) {
    return os << "CustomType(a=" << obj.a << ", b=" << obj.b << ")";
}


int main() {
    try {
        // Initialize logging system
        const auto &config = config::Configuration::getInstance();
        // common::logging::Logger::initializeWithLevel("debug");

        LOG_INFO("This is a number: {}", 42);
        LOG_WARN("This is a warning with a number: {}", 123);


        // Test logging with various types
        LOG_INFO("This is a number: {}", 42);
        LOG_WARN("This is a warning with value: {}", 100);
        LOG_ERROR("This is an error message");
        LOG_DEBUG("This is a debug message");
        LOG_TRACE("This is a trace message");
        LOG_CRITICAL("This is a critical message");

        // Test logging with a vector
        std::vector<int> vec = {1, 2, 3, 4, 5};
        LOG_INFO("Logging a vector: {}", vec);

        // Test general-purpose log macro
        LOG("info", "This is a general-purpose log with a number: {}", 7);
        LOG("warn", "This is a general-purpose warning");

        // Test logging with Eigen matrices
        Eigen::MatrixXd mat(3, 3);
        mat << 1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0;
        LOG_INFO("Logging an Eigen matrix:\n{}", mat);

        Eigen::MatrixXd transposed = mat.transpose();
        LOG_INFO("Logging a transposed Eigen matrix:\n{}", transposed);


        LOG_INFO("Logging a custom type: {}", CustomType());

        return 0;


        testDistanceEstimator("../task1/viewspace_images/obj_000020/rgb_0.png");
        testDistanceEstimator("../task1/viewspace_images/obj_000020/rgb_1.png");
        testDistanceEstimator("../task1/viewspace_images/obj_000020/rgb_2.png");
        testDistanceEstimator("../task1/viewspace_images/obj_000020/rgb_3.png");
        // return 0;
        // Load task parameters from configuration
        auto task_name = config.get<std::string>("task.name", "task1");
        auto objects = config.get<std::vector<std::string> >("task.objects", {"obj_000020"});
        int test_num = config.get<int>("task.test_num", 1);

        // Ensure test number is loaded or use a default value
        test_num = (test_num <= 0) ? 1 : test_num;

        spdlog::info("Starting task: {}, with {} objects and {} tests per object.", task_name, objects.size(),
                     test_num);

        // Get instance of TaskManager
        auto &task_manager = tasks::TaskManager::getInstance();

        // Execute task for each object
        for (const auto &object: objects) {
            spdlog::info("Executing task for object: {}", object);
            task_manager.execute(object, test_num);
        }

        spdlog::info("All tasks executed successfully.");
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        spdlog::critical("Critical error: {}", e.what());
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred." << std::endl;
        spdlog::critical("Unknown critical error.");
        return 1;
    }

    return 0;
}
