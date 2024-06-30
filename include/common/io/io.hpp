// File: common/io/io.hpp

#ifndef IO_HPP
#define IO_HPP

#include <fstream>
#include <string>
#include <filesystem>
#include "common/logging/logger.hpp"

namespace common::io {

    /**
     * @brief Checks if a file exists.
     *
     * @param file_path Path to the file as a string.
     * @return true if the file exists, false otherwise.
     */
    inline bool fileExists(const std::string &file_path) {
        std::filesystem::path path(file_path);
        LOG_DEBUG("Checking if file exists: {}", file_path);
        bool exists = std::ifstream(path).good();
        LOG_DEBUG(exists ? "File exists: {}" : "File does not exist: {}", file_path);
        return exists;
    }

    /**
     * @brief Checks if a directory exists.
     *
     * @param dir_path Path to the directory as a string.
     * @return true if the directory exists, false otherwise.
     */
    inline bool directoryExists(const std::string &dir_path) {
        std::filesystem::path path(dir_path);
        LOG_DEBUG("Checking if directory exists: {}", dir_path);
        bool exists = std::filesystem::exists(path) && std::filesystem::is_directory(path);
        LOG_DEBUG(exists ? "Directory exists: {}" : "Directory does not exist: {}", dir_path);
        return exists;
    }

    /**
     * @brief Creates a directory and all necessary parent directories.
     *
     * @param dir_path Path to the directory as a string.
     * @throws std::runtime_error if the directory could not be created.
     */
    inline void createDirectory(const std::string &dir_path) {
        std::filesystem::path path(dir_path);
        LOG_DEBUG("Creating directory: {}", dir_path);
        try {
            if (!std::filesystem::create_directories(path)) {
                LOG_ERROR("Could not create directory: {}", dir_path);
                throw std::runtime_error(fmt::format("Could not create directory: {}", dir_path));
            }
        } catch (const std::filesystem::filesystem_error &e) {
            LOG_ERROR("Filesystem error: {}", e.what());
            throw std::runtime_error(fmt::format("Could not create directory: {}", dir_path));
        }
        LOG_DEBUG("Directory created successfully: {}", dir_path);
    }
}

#endif // IO_HPP
