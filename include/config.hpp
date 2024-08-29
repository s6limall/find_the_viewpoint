//
// Created by ayush on 5/21/24.
//

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <json/json.h>
#include <iostream>
#include <fstream>

class Config {
public:
    static void initializeLogging(const std::string &logFilePath = "logs/logfile.log");
    static void setLoggingLevel(spdlog::level::level_enum level);
    Json::Value get_config();
};

#endif // CONFIG_HPP