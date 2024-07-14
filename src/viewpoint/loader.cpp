/*
// File: viewpoint/loader.cpp

#include "viewpoint/loader.hpp"

namespace viewpoint {

    Loader::Loader(std::string filepath):
        filepath_(std::move(filepath)) {
        LOG_DEBUG("Loader initialized with filepath: {}", filepath_);
    }

    std::vector<core::View> Loader::provision() {
        LOG_INFO("Loading viewpoints from file: {}", filepath_);
        std::vector<core::View> views;

        if (!common::io::fileExists(filepath_)) {
            LOG_ERROR("File does not exist: {}", filepath_);
            throw std::runtime_error("File does not exist: " + filepath_);
        }

        std::ifstream file(filepath_);
        Eigen::Vector3d position;
        while (file >> position(0) >> position(1) >> position(2)) {
            core::View view;
            //view.computePose(position, Eigen::Vector3f(0, 0, 0));
            view.computePose(position.normalized() * 3.0f, Eigen::Vector3d(0, 0, 0));
            views.push_back(view);
        }
        LOG_INFO("Loaded {} viewpoints from file", views.size());
        return views;
    }

}
*/
