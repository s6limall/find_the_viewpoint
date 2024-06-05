//
// Created by ayush on 5/21/24.
//

#include "../include/tasks.hpp"

#include <iostream>
#include <fstream>
#include <fmt/format.h>

#include "../include/core/perception.hpp"
#include "../include/core/simulator.hpp"


void Tasks::execute(const std::string& taskName, const std::string& object_name, int test_num) {
    if (taskName == "task1") {
        task1(object_name, test_num);
    }
    // Additional tasks
}

void Tasks::task1(const std::string& object_name, int test_num) {
    auto* perception_simulator = new Perception("../3d_models/" + object_name + ".ply");

    std::vector<View> view_space;
    std::ifstream fin("../view_space/5.txt");
    if (!fin.is_open()) {
        cout << "Open file failed!" << endl;
        exit(1);
    }
    Eigen::Vector3d position;
    while (fin >> position(0) >> position(1) >> position(2)) {
        position = position.normalized();
        View view;
        view.computePoseFromPositionAndObjectCenter(position * 3.0, Eigen::Vector3d(1e-100, 1e-100, 1e-100));
        view_space.push_back(view);
    }
    fin.close();

    for (size_t i = 0; i < view_space.size(); ++i) {
        perception_simulator->render(view_space[i], "../task1/viewspace_images/" + object_name + "/rgb_" + fmt::to_string(i) + ".png");
    }

    std::set<int> selected_view_indices;
    ResultsLogger results_logger;

    for (int test_id = 0; test_id < test_num; ++test_id) {
        int index;
        while (true) {
            index = rand() % view_space.size();
            if (selected_view_indices.find(index) == selected_view_indices.end()) {
                selected_view_indices.insert(index);
                break;
            }
        }

        cout << "Select view " << index << " for test " << test_id << endl;
        View target_view = view_space[index];
        cv::Mat target_image = cv::imread("../task1/viewspace_images/" + object_name + "/rgb_" + fmt::to_string(index) + ".png");

        // Save the target image separately for viewing (to debug/verify)
        cv::imwrite("../task1/selected_views/" + object_name + "/target_image_test_" + fmt::to_string(test_id) + ".png", target_image);

        Simulator simulator(perception_simulator, target_image, view_space, results_logger, test_id);
        simulator.loop();

        std::ofstream fout("../task1/selected_views/" + object_name + "/test_" + fmt::to_string(test_id) + ".txt");
        fout << simulator.selected_views.size() << endl;
        for (const auto& selected_view : simulator.selected_views) {
            fout << selected_view.pose_6d << endl;
        }
        fout.close();
    }

    results_logger.saveResults("../task1/selected_views/" + object_name + "/results.log");

    delete perception_simulator;
}
