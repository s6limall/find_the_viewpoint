// File: core/simulator.cpp

#include "core/simulator.hpp"
#include "processing/image_processor.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>

using namespace std;

namespace core {
    Simulator::Simulator(std::shared_ptr<Perception> perception_simulator, cv::Mat target_image,
                         std::vector<View> view_space) :
        perception_simulator_(std::move(perception_simulator)),
        target_image_(std::move(target_image)),
        view_space_(std::move(view_space)) {
    }


    Simulator::~Simulator() {
        spdlog::debug("Simulator destroyed successfully!");
    }

    // Renders an image from the specified view.
    cv::Mat Simulator::renderViewImage(View view) {
        std::string image_save_path = "../tmp/rgb.png";
        perception_simulator_->render(view.getPose(), image_save_path); // Use camera pose from View.
        cv::Mat rendered_image = cv::imread(image_save_path);
        remove(image_save_path.c_str());
        return rendered_image;
    }


    // Determines if the rendered image matches the target image.
    std::pair<bool, size_t> Simulator::isTarget(View view) {
        cv::Mat rendered_image = renderViewImage(view);
        return processing::image::ImageProcessor::compareImages(rendered_image, target_image_);
        // Compare rendered image with target.
    }

    // Finds the next best view based on similarity to the target image.
    View Simulator::searchNextView() {
        size_t max_good_matches = 0;
        View best_view;
        bool found_best_view = false;

        for (const auto &view: view_space_) {
            if (std::any_of(selected_views_.begin(), selected_views_.end(), [&](const View &v) {
                return view.getPose().isApprox(v.getPose()); // Check for similar poses.
            }))
                continue;

            cv::Mat rendered_image = renderViewImage(view);
            auto [result, good_matches] =
                    processing::image::ImageProcessor::compareImages(rendered_image, target_image_);
            // Compare images.

            spdlog::info("View {}: {} good matches", &view - &view_space_[0], good_matches);

            if (good_matches > max_good_matches) {
                max_good_matches = good_matches;
                best_view = std::move(view);;
                found_best_view = true;
            }
        }

        if (!found_best_view) {
            if (!selected_views_.empty()) {
                spdlog::warn("No good matches found, returning the first selected view.");
                return selected_views_[0];
            }
            spdlog::error("No available views to select from");
            throw std::runtime_error("No available views to select from");
        }

        spdlog::info("Best view found with {} good matches", max_good_matches);
        return best_view;
    }

    // Main loop to find the target view.
    void Simulator::loop() {
        size_t good_matches = 0;
        int count = 0;
        do {
            spdlog::info("Simulator loop iteration {}", count++);
            View next_view = searchNextView();
            selected_views_.push_back(next_view);
            if (auto [target_found, matches] = isTarget(next_view); target_found) {
                good_matches = matches;
                break;
            }
        } while (true);


        cout << "Found the target!" << endl;
    }
}
