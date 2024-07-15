// File: common/logging/logger.hpp

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <memory>
#include <string>
#include <mutex>
#include <filesystem>
#include <unordered_map>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/fmt/ostr.h>

#include "common/logging/logger.hpp"
// #include "common/formatting/fmt_camera_parameters.hpp"
#include "common/formatting/fmt_vector.hpp"
#include "common/formatting/fmt_view.hpp"
#include "common/formatting/fmt_eigen.hpp"
#include "common/formatting/fmt_cv.hpp"

// NOTE: Logger MUST NOT depend on config::Configuration, as it will cause a circular dependency
namespace common::logging {

    class Logger {
    public:
        // Delete copy constructor and assignment operator
        Logger(const Logger &) = delete;

        Logger &operator=(const Logger &) = delete;

        ~Logger() = default;

        template<typename... Args>
        static void log(spdlog::level::level_enum level, const char *file, int line, const char *func, const char *fmt,
                        Args &&... args);

        static void setLogLevel(const std::string &level);

        static void setPattern(const std::string &pattern);

        static spdlog::level::level_enum getLogLevel(const std::string &level);

        static std::shared_ptr<spdlog::logger> getLogger();

    private:
        static std::shared_ptr<spdlog::logger> logger_;
        static std::string pattern_;
        static spdlog::level::level_enum level_;
        static std::once_flag init_flag_;

        static void init(const std::string &log_directory = "./logs",
                         const std::string &log_filename = "application.log",
                         const std::string &log_level = "debug",
                         const std::string &pattern = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [%n] [%s:%# %!] %v");

        static void initialize(const std::string &log_directory, const std::string &log_filename);
    };

#define LOG(level, fmt, ...) common::logging::Logger::log(common::logging::Logger::getLogLevel(level), __FILE__, __LINE__, __FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_TRACE(fmt, ...) LOG("trace", fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) LOG("debug", fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) LOG("info", fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) LOG("warn", fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) LOG("error", fmt, ##__VA_ARGS__)
#define LOG_CRITICAL(fmt, ...) LOG("critical", fmt, ##__VA_ARGS__)

    // Template and inline method definitions
    template<typename... Args>
    void Logger::log(spdlog::level::level_enum level, const char *file, int line, const char *func, const char *fmt,
                     Args &&... args) {
        std::call_once(init_flag_, []() { init(); });
        if (logger_) {
            spdlog::source_loc source{file, line, func};
            if constexpr (sizeof...(args) > 0) {
                logger_->log(source, level, fmt, std::forward<Args>(args)...);
            } else {
                logger_->log(source, level, fmt);
            }
        } else {
            std::cerr << "Logger not initialized!" << std::endl;
        }
    }


}

#endif // LOGGER_HPP
