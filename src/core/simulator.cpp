//
// Created by ayush on 5/21/24.
//

#include "../../include/core/simulator.hpp"
#include "../../include/processing/image.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>

using namespace std;

Simulator::Simulator(Perception* _perception_simulator, cv::Mat _target_image, vector<View> _view_space) {
    perception_simulator = _perception_simulator;
    target_image = _target_image;
    view_space = _view_space;
}

Simulator::~Simulator() {
    cout << "Simulator destruct successfully!" << endl;
}

cv::Mat Simulator::renderViewImage(View view) {
    string image_save_path = "../tmp/rgb.png";
    perception_simulator->render(view, image_save_path);
    cv::Mat rendered_image = cv::imread(image_save_path);
    remove(image_save_path.c_str());
    return rendered_image;
}

bool Simulator::isTarget(View view) {
    cv::Mat rendered_image = renderViewImage(view);
    return compareImages(rendered_image, target_image);
}

View Simulator::searchNextView() {
    size_t maxGoodMatches = 0;
    View bestView;
    bool foundBestView = false;

    for (const auto &view : view_space) {
        if (std::any_of(selected_views.begin(), selected_views.end(), [&](const View &v) {
            return view.pose_6d.isApprox(v.pose_6d);
        }))
            continue;

        cv::Mat renderedImage = renderViewImage(view);
        size_t goodMatches = computeSIFTMatches(target_image, renderedImage);

        spdlog::info("View {}: {} good matches", &view - &view_space[0], goodMatches);

        if (goodMatches > maxGoodMatches) {
            maxGoodMatches = goodMatches;
            bestView = view;
            foundBestView = true;
        }
    }

    if (!foundBestView) {
        if (!selected_views.empty()) {
            spdlog::warn("No good matches found, returning the first selected view.");
            return selected_views[0];
        }
        spdlog::error("No available views to select from");
        throw std::runtime_error("No available views to select from");
    }

    spdlog::info("Best view found with {} good matches", maxGoodMatches);
    return bestView;
}


void Simulator::loop() {
    View next_view = searchNextView();
    selected_views.push_back(next_view);

    while (!isTarget(next_view)) {
        next_view = searchNextView();
        selected_views.push_back(next_view);
    }

    cout << "Find the target!" << endl;
}
