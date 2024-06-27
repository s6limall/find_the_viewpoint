//
// Created by ayush on 5/21/24.
//

#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "perception.hpp"
#include "view.hpp"


namespace core {
    // Manages simulation of rendering views from different camera positions.
    class Simulator {
    public:
        Simulator(std::shared_ptr<Perception> perception_simulator, cv::Mat target_image, std::vector<View> view_space);

        ~Simulator();

        // Renders an image from the specified view.
        cv::Mat renderViewImage(View view);

        // Determines if the rendered image matches the target image.
        std::pair<bool, size_t> isTarget(View view);

        // Finds the next best view based on similarity to the target image.
        View searchNextView();

        // Main loop to find the target view.
        void loop();

    private:
        std::shared_ptr<Perception> perception_simulator_; // Simulator for perception.
        cv::Mat target_image_; // Target image to match.
        std::vector<View> view_space_; // List of potential views.
        std::vector<View> selected_views_; // List of selected views.
        // std::vector<cv::Mat> rendered_images; // Rendered images.
    };
}

#endif // SIMULATOR_HPP
