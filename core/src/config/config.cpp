#include "../../include/config.hpp"
#include <iostream>
#include <fstream>
#include <json/json.h>
#include <unistd.h> // For getcwd()

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

Json::Value Config::get_config(){
	// Print the current working directory
	char cwd[1024];
	if (getcwd(cwd, sizeof(cwd)) != NULL) {
		std::cout << "Current working directory: " << cwd << std::endl;
	} else {
		std::cerr << "getcwd() error" << std::endl;
	}

	std::ifstream file("cfg/default.json");
	if (!file.is_open()) {
		std::cerr << "Unable to open file: cfg/default.json" << std::endl;
		// Check if the file exists
		std::ifstream testFile("cfg/default.json");
		if (!testFile) {
			std::cerr << "File does not exist." << std::endl;
		} else {
			std::cerr << "File exists but cannot be opened." << std::endl;
		}
		return Json::Value(); // Return an empty Json::Value object on failure
	}

	// Parse the JSON file
	Json::Value cfg;
	try {
		file >> cfg;
	} catch (const Json::RuntimeError& e) {
		std::cerr << "JSON parsing error: " << e.what() << std::endl;
		file.close();
		return Json::Value(); // Return an empty Json::Value object on parsing failure
	}

	// Close the file
	file.close();
	return cfg;
}