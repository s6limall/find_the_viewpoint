// Fil: commong/logging/logger.hpp

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/fmt/ostr.h>
#include <memory>
#include <string>
#include <mutex>
#include <unordered_map>

// Include custom formatters
#include "common/formatting/fmt_vector.hpp"
#include "common/formatting/fmt_eigen.hpp"
// #include "common/formatting/fmt_fallback.hpp"

namespace logger {

    class Logger {
    public:
        static void init(const std::string &log_directory = "./logs",
                         const std::string &log_filename = "application.log");

        template<typename... Args>
        static void log(spdlog::level::level_enum level, const char *file, int line, const char *func, const char *fmt,
                        Args &&... args);

        static spdlog::level::level_enum getLogLevel(const std::string &level);

    private:
        static std::shared_ptr<spdlog::logger> logger_instance;
        static std::once_flag init_flag;
    };

#define LOG(level, fmt, ...) logger::Logger::log(logger::Logger::getLogLevel(level), __FILE__, __LINE__, __FUNCTION__, fmt, __VA_ARGS__)
#define LOG_TRACE(fmt, ...) LOG("trace", fmt, __VA_ARGS__)
#define LOG_DEBUG(fmt, ...) LOG("debug", fmt, __VA_ARGS__)
#define LOG_INFO(fmt, ...) LOG("info", fmt, __VA_ARGS__)
#define LOG_WARN(fmt, ...) LOG("warn", fmt, __VA_ARGS__)
#define LOG_ERROR(fmt, ...) LOG("error", fmt, __VA_ARGS__)
#define LOG_CRITICAL(fmt, ...) LOG("critical", fmt, __VA_ARGS__)

    // Template and inline method definitions
    template<typename... Args>
    void Logger::log(spdlog::level::level_enum level, const char *file, int line, const char *func, const char *fmt,
                     Args &&... args) {
        std::call_once(init_flag, []() { init(); });
        if (logger_instance) {
            logger_instance->log(level, "[{}:{}:{}] " + std::string(fmt), file, line, func,
                                 std::forward<Args>(args)...);
        }
    }

}

#endif // LOGGER_HPP
