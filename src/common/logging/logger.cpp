// File: common/logging/logger.cpp

#include "common/logging/logger.hpp"
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <filesystem>

namespace logger {

    std::shared_ptr<spdlog::logger> Logger::logger_instance = nullptr;
    std::once_flag Logger::init_flag;

    void Logger::init(const std::string &log_directory, const std::string &log_filename) {
        if (!std::filesystem::exists(log_directory)) {
            std::filesystem::create_directories(log_directory);
        }
        std::vector<spdlog::sink_ptr> sinks;
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        sinks.push_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_directory + "/" + log_filename, true));

        logger_instance = std::make_shared<spdlog::logger>("logger", sinks.begin(), sinks.end());
        spdlog::register_logger(logger_instance);
        logger_instance->set_level(spdlog::level::trace);
        logger_instance->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%t] [%l] [%s:%#] [%!] %v");
    }

    spdlog::level::level_enum Logger::getLogLevel(const std::string &level) {
        static const std::unordered_map<std::string, spdlog::level::level_enum> level_map = {
                {"trace", spdlog::level::trace},
                {"debug", spdlog::level::debug},
                {"info", spdlog::level::info},
                {"warn", spdlog::level::warn},
                {"error", spdlog::level::err},
                {"critical", spdlog::level::critical}
        };
        auto it = level_map.find(level);
        return it != level_map.end() ? it->second : spdlog::level::info; // Default to info level
    }
}
