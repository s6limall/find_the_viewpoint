// File: common/utilities/file_utils.hpp

#ifndef FILE_UTILS_HPP
#define FILE_UTILS_HPP

#include <string>

namespace common {
    namespace utilities {
        class FileUtils {
        public:
            // Ensures the specified directory exists, creates it if necessary
            static void ensureDirectoryExists(const std::string &path);

            // Reads a file and returns its content as a string
            static std::string readFile(const std::string& file_path);

            // Writes a string to a file
            static void writeStringToFile(const std::string &filepath, const std::string &content);

            // Checks if a file exists
            static bool fileExists(const std::string &filepath);
        };
    }
}

#endif // FILE_UTILS_HPP
