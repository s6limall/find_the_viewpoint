//
// Created by ayush on 5/21/24.
//

#include "../../include/core/simulator.hpp"
#include "../../include/processing/image.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>

using namespace std;

Simulator::Simulator(Perception *_perception_simulator, cv::Mat _target_image, std::vector<View> _view_space,
                     ResultsLogger &_results_logger, int _test_id)
    : perception_simulator(_perception_simulator), target_image(_target_image), view_space(_view_space),
      results_logger(_results_logger), test_id(_test_id) {
}

Simulator::~Simulator() {
    cout << "Simulator destruct successfully!" << endl;
}

cv::Mat Simulator::renderViewImage(View view) {
    std::string image_save_path = "../tmp/rgb.png";
    perception_simulator->render(view, image_save_path);
    cv::Mat rendered_image = cv::imread(image_save_path);
    remove(image_save_path.c_str());
    return rendered_image;
}

std::pair<bool, size_t> Simulator::isTarget(View view) {
    cv::Mat rendered_image = renderViewImage(view);
    return compareImages(rendered_image, target_image);
}

View Simulator::searchNextView() {
    size_t max_good_matches = 0;
    View best_view;
    bool found_best_view = false;

    for (const auto &view: view_space) {
        if (std::any_of(selected_views.begin(), selected_views.end(), [&](const View &v) {
            return view.pose_6d.isApprox(v.pose_6d);
        }))
            continue;

        cv::Mat rendered_image = renderViewImage(view);
        auto [result, good_matches] = compareImages(rendered_image, target_image);

        spdlog::info("View {}: {} good matches", &view - &view_space[0], good_matches);

        if (good_matches > max_good_matches) {
            max_good_matches = good_matches;
            best_view = view;
            found_best_view = true;
        }
    }

    if (!found_best_view) {
        if (!selected_views.empty()) {
            spdlog::warn("No good matches found, returning the first selected view.");
            return selected_views[0];
        }
        spdlog::error("No available views to select from");
        throw std::runtime_error("No available views to select from");
    }

    spdlog::info("Best view found with {} good matches", max_good_matches);
    return best_view;
}


void Simulator::loop() {
    size_t good_matches = 0;

    do {
        View next_view = searchNextView();
        selected_views.push_back(next_view);
        if (auto [target_found, matches] = isTarget(next_view); target_found) {
            good_matches = matches;
            break; // Exit when the target is found
        }
    } while (true);

    // Log the results after finding the target
    ResultsLogger::TestResult result(test_id, selected_views.size() - 1, good_matches, selected_views);
    results_logger.addResult(result);

    cout << "Found the target!" << endl;
}
