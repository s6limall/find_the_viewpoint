// File: common/io/io.hpp

#ifndef IO_HPP
#define IO_HPP

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "common/logging/logger.hpp"

namespace common::io {

    namespace fs = std::filesystem;

    /**
     * @brief Checks if a file exists.
     *
     * @param file_path Path to the file.
     * @return true if the file exists, false otherwise.
     */
    [[nodiscard]] inline bool fileExists(const fs::path &file_path) noexcept {
        std::error_code ec;
        if (fs::is_regular_file(file_path, ec)) {
            LOG_DEBUG("File exists: {}", file_path.string());
            return true;
        }
        LOG_DEBUG("File does not exist or is not accessible: {}", file_path.string());
        return false;
    }

    /**
     * @brief Checks if a directory exists.
     *
     * @param dir_path Path to the directory.
     * @return true if the directory exists, false otherwise.
     */
    [[nodiscard]] inline bool directoryExists(const fs::path &dir_path) noexcept {
        std::error_code ec;
        if (fs::is_directory(dir_path, ec)) {
            LOG_DEBUG("Directory exists: {}", dir_path.string());
            return true;
        }
        LOG_DEBUG("Directory does not exist or is not accessible: {}", dir_path.string());
        return false;
    }

    /**
     * @brief Creates a directory and all necessary parent directories.
     *
     * @param dir_path Path to the directory.
     * @throws std::invalid_argument if the path is empty or just a filename.
     * @throws std::runtime_error if the directory could not be created.
     */
    inline void createDirectory(const fs::path &dir_path) {
        if (dir_path.empty()) {
            LOG_ERROR("Empty directory path provided");
            throw std::invalid_argument("Empty directory path");
        }

        if (dir_path.has_filename() && !dir_path.has_parent_path()) {
            LOG_ERROR("Invalid directory path: {}. Appears to be a filename.", dir_path.string());
            throw std::invalid_argument(
                    fmt::format("Invalid directory path: {}. Appears to be a filename.", dir_path.string()));
        }

        LOG_DEBUG("Creating directory: {}", dir_path.string());
        std::error_code ec;
        if (!fs::create_directories(dir_path, ec)) {
            if (ec) {
                LOG_ERROR("Could not create directory: {}. Error: {}", dir_path.string(), ec.message());
                throw std::runtime_error(
                        fmt::format("Could not create directory: {}. Error: {}", dir_path.string(), ec.message()));
            } else {
                LOG_DEBUG("Directory already exists: {}", dir_path.string());
            }
        } else {
            LOG_DEBUG("Directory created successfully: {}", dir_path.string());
        }
    }

    // ... (other functions remain the same)

    /**
     * @brief Lists full paths of files in a directory with a specific extension.
     *
     * @param dir_path Path to the directory.
     * @param extension File extension to filter by (including the dot, e.g., ".ply").
     * @return std::vector<fs::path> List of full file paths in the directory with the specified extension.
     * @throws std::invalid_argument if the path is empty or not a directory.
     * @throws std::runtime_error if the directory could not be read.
     */
    [[nodiscard]] inline std::vector<fs::path> filesByExtension(const fs::path &dir_path, std::string_view extension) {
        if (dir_path.empty()) {
            LOG_ERROR("Empty directory path provided");
            throw std::invalid_argument("Empty directory path");
        }

        if (!fs::is_directory(dir_path)) {
            LOG_ERROR("Path is not a directory: {}", dir_path.string());
            throw std::invalid_argument(fmt::format("Path is not a directory: {}", dir_path.string()));
        }

        LOG_DEBUG("Listing files with extension {} in directory: {}", extension, dir_path.string());
        std::vector<fs::path> matching_files;
        std::error_code ec;
        for (const auto &entry: fs::directory_iterator(dir_path, ec)) {
            if (ec) {
                LOG_ERROR("Error while iterating directory: {}. Error: {}", dir_path.string(), ec.message());
                throw std::runtime_error(
                        fmt::format("Error while iterating directory: {}. Error: {}", dir_path.string(), ec.message()));
            }
            if (entry.is_regular_file() && entry.path().extension() == extension) {
                matching_files.push_back(entry.path());
            }
        }

        std::ranges::sort(matching_files);
        LOG_DEBUG("Found {} files with extension {} in directory: {}", matching_files.size(), extension,
                  dir_path.string());
        return matching_files;
    }

    /**
     * @brief Lists filenames of files in a directory with a specific extension.
     *
     * @param dir_path Path to the directory.
     * @param extension File extension to filter by (including the dot, e.g., ".ply").
     * @return std::vector<std::string> List of filenames in the directory with the specified extension.
     * @throws std::invalid_argument if the path is empty or not a directory.
     * @throws std::runtime_error if the directory could not be read.
     */
    [[nodiscard]] inline std::vector<std::string> filenamesByExtension(const fs::path &dir_path,
                                                                       std::string_view extension) {
        if (dir_path.empty()) {
            LOG_ERROR("Empty directory path provided");
            throw std::invalid_argument("Empty directory path");
        }

        if (!fs::is_directory(dir_path)) {
            LOG_ERROR("Path is not a directory: {}", dir_path.string());
            throw std::invalid_argument(fmt::format("Path is not a directory: {}", dir_path.string()));
        }

        LOG_DEBUG("Listing filenames with extension {} in directory: {}", extension, dir_path.string());
        std::vector<std::string> matching_filenames;
        std::error_code ec;
        for (const auto &entry: fs::directory_iterator(dir_path, ec)) {
            if (ec) {
                LOG_ERROR("Error while iterating directory: {}. Error: {}", dir_path.string(), ec.message());
                throw std::runtime_error(
                        fmt::format("Error while iterating directory: {}. Error: {}", dir_path.string(), ec.message()));
            }
            if (entry.is_regular_file() && entry.path().extension() == extension) {
                matching_filenames.push_back(entry.path().filename().string());
            }
        }

        std::ranges::sort(matching_filenames);
        LOG_DEBUG("Found {} files with extension {} in directory: {}", matching_filenames.size(), extension,
                  dir_path.string());
        return matching_filenames;
    }

} // namespace common::io

#endif // IO_HPP
