//
// Created by ayush on 5/21/24.
//

#include "../include/tasks.hpp"

#include <iostream>
#include <fstream>
#include <random>
#include <fmt/format.h>

#include "../include/core/perception.hpp"
#include "../include/core/simulator.hpp"


// Helper function to load viewpoints from a file.
// This function reads viewpoints from a file, computes their poses relative to an object center, and returns them.
std::vector<View> loadViewpoints(const std::string &filepath) {
    std::vector<View> views;
    std::ifstream fin(filepath);
    if (!fin) {
        throw std::runtime_error("Failed to open view file: " + filepath);
    }

    Eigen::Vector3f position;
    while (fin >> position(0) >> position(1) >> position(2)) {
        View view;
        view.computePoseFromPositionAndObjectCenter(position.normalized() * 3.0f, Eigen::Vector3f(0, 0, 0));
        views.push_back(view);
    }
    return views;
}

// Function to initialize the Perception object with the model path.
// This function creates a new Perception object for rendering the 3D model.
std::unique_ptr<Perception> initializePerception(const std::string &model_path) {
    return std::make_unique<Perception>(model_path);
}

// The function performs a viewpoint evaluation task based on provided parameters.
// It loads viewpoints, initializes a perception simulator, and performs multiple tests to render views and save results.
void Tasks::performViewpointEvaluation(const std::string &taskName, const std::string &object_name, int test_num,
                                       const std::string &view_file_path) {
    // Define the base directory for tasks
    std::string task_directory = "../" + taskName; // Assuming task directories are named after the tasks

    // Initialize the Perception Simulator with the object model
    auto perception_simulator = initializePerception("../3d_models/" + object_name + ".ply");

    // Load viewpoints from the specified file
    std::vector<View> view_space = loadViewpoints(view_file_path);

    std::cerr << "Loaded " << view_space.size() << " viewpoints from " << view_file_path << std::endl;

    // Loop over each test
    for (int test_id = 0; test_id < test_num; ++test_id) {
        int index = rand() % view_space.size(); // Select a random view
        std::cerr << "Select view " << index << " for test " << test_id << std::endl;

        // Define the image path for the rendered view
        std::string image_path = task_directory + "/viewspace_images/" + object_name + "/rgb_" + std::to_string(index) +
                                 ".png";

        // Render the view and save the image
        perception_simulator->render(view_space[index].getCameraPose(), image_path);
        cv::Mat target_image = cv::imread(image_path);

        // Save the target image for debugging purposes
        cv::imwrite(
            task_directory + "/selected_views/" + object_name + "/target_image_test_" + std::to_string(test_id) +
            ".png", target_image);

        // Perform the simulation with the selected view and the target image
        Simulator simulator(perception_simulator.get(), target_image, view_space);
        simulator.loop();

        // Save the results of the simulation to a text file
        std::ofstream fout(
            task_directory + "/selected_views/" + object_name + "/test_" + std::to_string(test_id) + ".txt");
        fout << simulator.selected_views.size() << std::endl;
        for (const auto &selected_view: simulator.selected_views) {
            fout << selected_view.getCameraPose() << std::endl;
        }
    }
}
