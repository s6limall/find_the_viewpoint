// File: common/utilities/file_utils.cpp

#include "common/utilities/file_utils.hpp"
#include <fstream>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace common::utilities {

    // Ensures the specified directory exists, creating it if it does not exist
    void FileUtils::ensureDirectoryExists(const std::string &path) {
        if (!std::filesystem::exists(path)) {
            std::filesystem::create_directories(path);
            spdlog::info("Created directory: {}", path);
        }
    }

    // Reads the entire contents of a file into a string
    std::string FileUtils::readFile(const std::string& file_path) {
        std::ifstream file(file_path);
        if (!file) {
            spdlog::error("Failed to open file: {}", file_path);
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    // Writes the provided content to a file
    void FileUtils::writeStringToFile(const std::string &filepath, const std::string &content) {
        std::ofstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("Failed to open file: {}", filepath);
            throw std::runtime_error("Failed to open file: " + filepath);
        }

        // Write content to the file
        file << content;
        file.close();
    }

    // Checks if a file exists at the specified path
    bool FileUtils::fileExists(const std::string &filepath) {
        return std::filesystem::exists(filepath);
    }

}

