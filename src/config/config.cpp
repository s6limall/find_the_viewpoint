#include "../../include/config.hpp"
#include <iostream>

void Config::initializeLogging(const std::string &logFilePath) {
    try {
        auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        consoleSink->set_level(spdlog::level::trace);

        // auto fileSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(logFilePath, true);
        // fileSink->set_level(spdlog::level::trace);

        // std::vector<spdlog::sink_ptr> sinks{consoleSink, fileSink};

        auto logger = std::make_shared<spdlog::logger>("console", consoleSink);
        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::trace); // Set global log level to trace
        spdlog::flush_on(spdlog::level::trace); // Flush log on every trace level message
        spdlog::set_pattern("[%T] [%l] %v"); // Set pattern to only show time
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
