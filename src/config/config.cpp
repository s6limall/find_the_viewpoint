//
// Created by ayush on 5/21/24.
//

#include "../../include/config/config.hpp"

#include <iostream>
#include <spdlog/spdlog.h>

// Initialize static member variables
const std::string Config::Paths::logsDirectory = "logs/";
const std::string Config::Paths::viewSpaceFile = "../view_space/5.txt";
const std::string Config::Paths::modelDirectory = "../3d_models/";
const std::string Config::Paths::viewSpaceImagesDirectory = "../task1/viewspace_images/";
const std::string Config::Paths::selectedViewsDirectory = "../task1/selected_views/";

const double Config::CameraConfig::width = 640;
const double Config::CameraConfig::height = 480;
const double Config::CameraConfig::fov_x = 0.95;
const double Config::CameraConfig::fov_y = 0.75;

void Config::initializeLogging(const std::string &logFilePath) {
    try {
        auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFilePath, true);

        consoleSink->set_level(spdlog::level::trace);
        fileSink->set_level(spdlog::level::trace);

        std::vector<spdlog::sink_ptr> sinks{consoleSink, fileSink};
        auto logger = std::make_shared<spdlog::logger>("multi_sink", sinks.begin(), sinks.end());
        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::trace); // Set global log level to trace
        spdlog::flush_on(spdlog::level::trace); // Flush log on every trace level message
        spdlog::info("Logging initialized");
    } catch (const spdlog::spdlog_ex &ex) {
        std::cerr << "Log initialization failed: " << ex.what() << std::endl;
    }
}

void Config::setLoggingLevel(spdlog::level::level_enum level) {
    spdlog::set_level(level);
    for (auto &sink: spdlog::default_logger()->sinks()) {
        sink->set_level(level);
    }
    spdlog::info("Logging level set to {}", spdlog::level::to_string_view(level));
}
