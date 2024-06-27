// viewpoint/loader.cpp

#include <fstream>
#include <utility>
#include <spdlog/spdlog.h>

#include "core/view.hpp"
#include "viewpoint/loader.hpp"
#include "common/utilities/file_utils.hpp"

namespace viewpoint {

    Loader::Loader(std::string filepath) :
        filepath_(std::move(filepath)) {
        spdlog::debug("Loader initialized with filepath: {}", filepath_);
    }

    std::vector<core::View> Loader::provision() {
        spdlog::info("Loading viewpoints from file: {}", filepath_);
        std::vector<core::View> views;

        if (!common::utilities::FileUtils::fileExists(filepath_)) {
            spdlog::error("File does not exist: {}", filepath_);
            throw std::runtime_error("File does not exist: " + filepath_);
        }

        std::ifstream file(filepath_);
        Eigen::Vector3f position;
        while (file >> position(0) >> position(1) >> position(2)) {
            core::View view;
            //view.computePoseFromPositionAndObjectCenter(position, Eigen::Vector3f(0, 0, 0));
            view.computePoseFromPositionAndObjectCenter(position.normalized() * 3.0f, Eigen::Vector3f(0, 0, 0));
            views.push_back(view);
        }
        spdlog::info("Loaded {} viewpoints from file", views.size());
        return views;
    }

}
