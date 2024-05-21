//
// Created by ayush on 5/21/24.
//

#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "perception.hpp"
#include "view.hpp"

class Simulator {
public:
    Perception* perception_simulator;
    cv::Mat target_image;
    std::vector<View> view_space;
    std::vector<View> selected_views;
    std::vector<cv::Mat> rendered_images;

    Simulator(Perception* _perception_simulator, cv::Mat _target_image, std::vector<View> _view_space = std::vector<View>());
    ~Simulator();

    cv::Mat renderViewImage(View view);
    bool isTarget(View view);
    View searchNextView();
    void loop();
};

#endif // SIMULATOR_HPP
