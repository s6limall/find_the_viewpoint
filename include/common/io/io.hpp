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
     * @param file_path Path to the file as a string or string_view.
     * @return true if the file exists, false otherwise.
     */
    inline bool fileExists(std::string_view file_path) {
        if (const std::filesystem::path path(file_path); std::ifstream(path).good()) {
            LOG_DEBUG("File exists: {}", file_path);
            return true;
        }

        LOG_DEBUG("File does not exist: {}", file_path);
        return false;
    }

    /**
     * @brief Checks if a directory exists.
     *
     * @param dir_path Path to the directory as a string or string_view.
     * @return true if the directory exists, false otherwise.
     */
    inline bool directoryExists(std::string_view dir_path) {
        if (const std::filesystem::path path(dir_path); exists(path) && is_directory(path)) {
            LOG_DEBUG("Directory exists: {}", dir_path);
            return true;
        }

        LOG_DEBUG("Directory does not exist: {}", dir_path);
        return false;
    }


    /**
     * @brief Creates a directory and all necessary parent directories.
     *
     * @param dir_path Path to the directory as a string.
     * @throws std::runtime_error if the directory could not be created.
     */
    inline void createDirectory(const std::string &dir_path) {
        LOG_DEBUG("Creating directory: {}", dir_path);
        try {
            if (const std::filesystem::path path(dir_path); !create_directories(path)) {
                LOG_ERROR("Could not create directory: {}", dir_path);
                throw std::runtime_error(fmt::format("Could not create directory: {}", dir_path));
            }
        } catch (const std::filesystem::filesystem_error &e) {
            LOG_ERROR("Filesystem error: {}", e.what());
            throw std::runtime_error(fmt::format("Could not create directory: {}", dir_path));
        }
        LOG_DEBUG("Directory created successfully: {}", dir_path);
    }


    /**
     * @brief Deletes a file.
     *
     * @param file_path Path to the file as a string or string_view.
     * @throws std::runtime_error if the file could not be deleted.
     */
    inline void deleteFile(std::string_view file_path) {
        if (const std::filesystem::path path(file_path); !std::filesystem::remove(path)) {
            LOG_ERROR("Could not delete file: {}", file_path);
            throw std::runtime_error(fmt::format("Could not delete file: {}", file_path));
        }
        LOG_DEBUG("File deleted successfully: {}", file_path);
    }

    /**
     * @brief Lists files in a directory.
     *
     * @param dir_path Path to the directory as a string or string_view.
     * @return std::vector<std::string> List of file names in the directory.
     * @throws std::runtime_error if the directory could not be read.
     */
    inline std::vector<std::string> listFilesInDirectory(std::string_view dir_path) {
        if (const std::filesystem::path path(dir_path);
            std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
            LOG_DEBUG("Listing files in directory: {}", dir_path);
            std::vector<std::string> files;
            for (const auto &entry: std::filesystem::directory_iterator(path)) {
                if (entry.is_regular_file()) {
                    files.push_back(entry.path().string());
                }
            }
            LOG_DEBUG("Listed {} files in directory: {}", files.size(), dir_path);
            return files;
        } else {
            LOG_ERROR("Directory does not exist or is not a directory: {}", dir_path);
            throw std::runtime_error(fmt::format("Directory does not exist or is not a directory: {}", dir_path));
        }
    }

    /**
     * @brief Copies a file.
     *
     * @param source_path Path to the source file as a string or string_view.
     * @param destination_path Path to the destination file as a string or string_view.
     * @param overwrite Flag indicating whether to overwrite the destination file if it exists.
     * @throws std::runtime_error if the file could not be copied.
     */
    inline void copyFile(std::string_view source_path, std::string_view destination_path, bool overwrite = false) {
        const std::filesystem::copy_options options = overwrite
                                                          ? std::filesystem::copy_options::overwrite_existing
                                                          : std::filesystem::copy_options::none;

        try {
            std::filesystem::copy(source_path, destination_path, options);
            LOG_DEBUG("File copied successfully from {} to {}", source_path, destination_path);
        } catch (const std::filesystem::filesystem_error &e) {
            LOG_ERROR("Could not copy file: {}", e.what());
            throw std::runtime_error(fmt::format("Could not copy file: {}", e.what()));
        }
    }

    /**
     * @brief Moves a file.
     *
     * @param source_path Path to the source file as a string or string_view.
     * @param destination_path Path to the destination file as a string or string_view.
     * @throws std::runtime_error if the file could not be moved.
     */
    inline void moveFile(std::string_view source_path, std::string_view destination_path) {
        try {
            std::filesystem::rename(source_path, destination_path);
            LOG_DEBUG("File moved successfully from {} to {}", source_path, destination_path);
        } catch (const std::filesystem::filesystem_error &e) {
            LOG_ERROR("Could not move file: {}", e.what());
            throw std::runtime_error(fmt::format("Could not move file: {}", e.what()));
        }
    }
}

#endif // IO_HPP
