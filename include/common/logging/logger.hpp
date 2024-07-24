// File: common/logging/logger.hpp

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "common/logging/logger.hpp"
// #include "common/formatting/fmt_camera_parameters.hpp"
#include "common/formatting/fmt_cv.hpp"
#include "common/formatting/fmt_eigen.hpp"
#include "common/formatting/fmt_vector.hpp"
#include "common/formatting/fmt_view.hpp"

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
                        Args &&...args);

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
                         const std::string &log_filename = "application.log", const std::string &log_level = "debug",
                         const std::string &pattern = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [%n] [%s:%# %!] %v");

        static void initialize(const std::string &log_directory, const std::string &log_filename);
    };

#define LOG_(level, fmt, ...)                                                                                          \
    common::logging::Logger::log(common::logging::Logger::getLogLevel(level), __FILE__, __LINE__, __FUNCTION__, fmt,   \
                                 ##__VA_ARGS__)
#define LOG_TRACE(fmt, ...) LOG_("trace", fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) LOG_("debug", fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) LOG_("info", fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) LOG_("warn", fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) LOG_("error", fmt, ##__VA_ARGS__)
#define LOG_CRITICAL(fmt, ...) LOG_("critical", fmt, ##__VA_ARGS__)

    // Template and inline method definitions
    template<typename... Args>
    void Logger::log(spdlog::level::level_enum level, const char *file, int line, const char *func, const char *fmt,
                     Args &&...args) {
        std::call_once(init_flag_, []() { init(); });
        if (logger_) {
            spdlog::source_loc source{file, line, func};
            if constexpr (sizeof...(args) > 0) {
                auto formatted_string = fmt::vformat(fmt, fmt::make_format_args(args...));
                logger_->log(source, level, formatted_string);
            } else {
                logger_->log(source, level, fmt);
            }
        } else {
            std::cerr << "Logger not initialized!" << std::endl;
        }
    }


} // namespace common::logging

#endif // LOGGER_HPP
