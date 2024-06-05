//
// Created by ayush on 5/21/24.
//

#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include "perception.hpp"
#include "view.hpp"

#include "../logging/results_logger.hpp"

class Simulator {
public:
    Perception* perception_simulator;
    cv::Mat target_image;
    std::vector<View> view_space;
    std::vector<View> selected_views;
    std::vector<cv::Mat> rendered_images;
    ResultsLogger& results_logger;
    int test_id;

    Simulator(Perception* _perception_simulator, cv::Mat _target_image, std::vector<View> _view_space, ResultsLogger& _results_logger, int _test_id);
    ~Simulator();

    cv::Mat renderViewImage(View view);

    std::pair<bool, size_t> isTarget(View view);

    View searchNextView();

    void loop();
};

#endif // SIMULATOR_HPP
