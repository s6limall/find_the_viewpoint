// File: common/logging/logger.cpp

#include "common/logging/logger.hpp"

namespace common::logging {

    std::shared_ptr<spdlog::logger> Logger::logger_ = nullptr;
    spdlog::level::level_enum Logger::level_ = spdlog::level::debug;
    std::string Logger::pattern_ = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [%n] [%s:%# %!] %v";
    std::once_flag Logger::init_flag_;

    void Logger::init(const std::string &log_directory, const std::string &log_filename, const std::string &log_level,
                      const std::string &pattern) {
        pattern_ = pattern;
        level_ = getLogLevel(log_level);
        initialize(log_directory, log_filename);
    }

    void Logger::initialize(const std::string &log_directory, const std::string &log_filename) {
        try {
            if (!std::filesystem::exists(log_directory)) {
                std::filesystem::create_directories(log_directory);
            }

            std::vector<spdlog::sink_ptr> sinks;
            sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
            sinks.push_back(
                    std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_directory + "/" + log_filename, true));

            logger_ = std::make_shared<spdlog::logger>("logger", sinks.begin(), sinks.end());
            logger_->set_level(level_);
            logger_->set_pattern(pattern_);

            spdlog::register_logger(logger_);
            spdlog::set_default_logger(logger_);
        } catch (const spdlog::spdlog_ex &ex) {
            std::cerr << "Log initialization failed: " << ex.what() << std::endl;
        }
    }

    void Logger::setLogLevel(const std::string &level) {
        std::call_once(init_flag_, []() { init(); });
        level_ = getLogLevel(level);
        if (logger_) {
            logger_->set_level(level_);
        }
    }

    void Logger::setPattern(const std::string &pattern) {
        std::call_once(init_flag_, []() { init(); });
        pattern_ = pattern;
        if (logger_) {
            logger_->set_pattern(pattern_);
        }
    }

    spdlog::level::level_enum Logger::getLogLevel(const std::string &level) {
        static const std::unordered_map<std::string_view, spdlog::level::level_enum> level_map = {
                {"trace", spdlog::level::trace},
                {"debug", spdlog::level::debug},
                {"info", spdlog::level::info},
                {"warn", spdlog::level::warn},
                {"error", spdlog::level::err},
                {"critical", spdlog::level::critical}
        };
        const auto iterator = level_map.find(level);
        return iterator != level_map.end() ? iterator->second : spdlog::level::debug;
    }

    std::shared_ptr<spdlog::logger> Logger::getLogger() {
        std::call_once(init_flag_, []() { init(); });
        return logger_;
    }

}
