//
// Created by ayush on 5/21/24.
//

#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "../include/config/config.hpp"
#include "../include/core/view.hpp"
#include "../include/core/perception.hpp"
#include "../include/core/simulator.hpp"
#include "../include/processing/image.hpp"

using namespace std;

void task1(const string& object_name, int test_num) {
    Perception* perception_simulator = new Perception("../3d_models/" + object_name + ".ply");

    vector<View> view_space;
    ifstream fin("../view_space/5.txt");
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
        perception_simulator->render(view_space[i], "../task1/viewspace_images/" + object_name + "/rgb_" + to_string(i) + ".png");
    }

    set<int> selected_view_indices;
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
        cv::Mat target_image = cv::imread("../task1/viewspace_images/" + object_name + "/rgb_" + to_string(index) + ".png");

        Simulator simulator(perception_simulator, target_image, view_space);
        simulator.loop();

        ofstream fout("../task1/selected_views/" + object_name + "/test_" + to_string(test_id) + ".txt");
        fout << simulator.selected_views.size() << endl;
        for (const auto& selected_view : simulator.selected_views) {
            fout << selected_view.pose_6d << endl;
        }
        fout.close();
    }

    delete perception_simulator;
}

int main() {
    Config::initializeLogging();
    Config::setLoggingLevel(spdlog::level::trace);

    srand(43);

    vector<string> objects = {"obj_000020"};
    int test_num = 5;

    for (const auto& object : objects) {
        task1(object, test_num);
    }

    return 0;
}
