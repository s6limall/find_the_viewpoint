//
// Created by ayush on 5/21/24.
//

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

class Config {
public:
    static void initializeLogging(const std::string &logFilePath = "logs/logfile.log");
    static void setLoggingLevel(spdlog::level::level_enum level);

    struct Paths {
        static const std::string logsDirectory;
        static const std::string viewSpaceFile;
        static const std::string modelDirectory;
        static const std::string viewSpaceImagesDirectory;
        static const std::string selectedViewsDirectory;
    };

    struct CameraConfig {
        static const double width;
        static const double height;
        static const double fov_x;
        static const double fov_y;
    };
};

#endif // CONFIG_HPP
