// File: config/configuration.hpp

#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <stdexcept>

#include "common/formatting/fmt_vector.hpp"

namespace config {

    class Configuration {
    public:
        // Disable copy constructor and assignment operator
        Configuration(const Configuration &) = delete;

        Configuration &operator=(const Configuration &) = delete;

        // Get the singleton instance with optional filename
        static Configuration &getInstance(const std::string &filename = "");

        // Check if a key exists in the configuration
        bool contains(const std::string &key) const;

        // Helper function to log entire configuration
        void logConfiguration() const;

        // Get a value of type T from the configuration
        template<typename T>
        T get(const std::string &key) const;

        // Get a value of type T from the configuration with a default value
        template<typename T>
        T get(const std::string &key, T default_value) const;

    private:
        explicit Configuration(const std::string &filename);

        ~Configuration() = default;

        // Load the entire configuration into a map
        void load(const YAML::Node &node, const std::string &prefix = "");


        // Helper function to split keys
        static std::vector<std::string> split(const std::string &str, char delimiter);\

        // Configuration map to hold key-value pairs
        std::unordered_map<std::string, YAML::Node> config_map_;

        static constexpr const char *default_filename_ = "configuration.yaml";

    };

    // Implementation of template methods

    template<typename T>
    T Configuration::get(const std::string &key) const {
        auto it = config_map_.find(key);
        if (it == config_map_.end()) {
            spdlog::warn("Key '{}' not found in configuration", key);
            throw std::runtime_error("Key not found in configuration");
        }
        try {
            return it->second.as<T>();
        } catch (const YAML::Exception &e) {
            spdlog::error("YAML parsing exception for key '{}': {}", key, e.what());
            throw std::runtime_error("YAML exception for key");
        } catch (const std::exception &e) {
            spdlog::error("Error fetching value for key '{}': {}", key, e.what());
            throw;
        }
    }

    template<typename T>
    T Configuration::get(const std::string &key, T default_value) const {
        try {
            return get<T>(key);
        } catch (const std::exception &e) {
            spdlog::info("Returning default value for key '{}': {}", key, default_value);
            return default_value;
        }
    }

    inline bool Configuration::contains(const std::string &key) const {
        return config_map_.find(key) != config_map_.end();
    }


}

#endif // CONFIGURATION_HPP
