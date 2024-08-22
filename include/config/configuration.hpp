// File: config/configuration.hpp

#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/logging/logger.hpp"

/*namespace YAML {
    template<>
    struct convert<std::string_view> {
        static Node encode(const std::string_view& rhs) {
            return Node(std::string(rhs));
        }

        static bool decode(const Node& node, std::string_view& rhs) {
            if (!node.IsScalar()) {
                return false;
            }
            rhs = node.Scalar();
            return true;
        }
    };
}*/


namespace config {

    class Configuration {
    public:
        // Delete copy constructor and assignment operator
        Configuration(const Configuration &) = delete;
        Configuration &operator=(const Configuration &) = delete;

        explicit Configuration(const std::string &filename);

        // Get the singleton instance with optional filename
        static Configuration &getInstance(const std::string &filename = "");

        // Check if a key exists in the configuration
        bool contains(const std::string &key) const;

        // Helper function to log entire configuration
        void show() const;

        // Get a value of type T from the configuration
        template<typename T>
        std::optional<T> get(const std::string &key) const;

        // Get a value of type T from the configuration with a default value
        template<typename T>
        T get(const std::string &key, T default_value) const;

        // Specialization to handle const char* as std::string
        std::string get(const std::string &key, const char *default_value) const;

    private:
        std::unordered_map<std::string, YAML::Node> config_map_;
        static constexpr std::string_view default_filename_ = "configuration.yaml";
        static std::shared_ptr<Configuration> instance_;
        static std::once_flag init_flag_;

        // Load the entire configuration into a map
        void load(const YAML::Node &node, const std::string &prefix = "");

        // Helper function to split keys
        static std::vector<std::string> split(const std::string &str, char delimiter);
    };

    // Template definitions
    template<typename T>
    std::optional<T> Configuration::get(const std::string &key) const {
        const auto it = config_map_.find(key);
        if (it == config_map_.end()) {
            LOG_WARN("Key '{}' not found in configuration", key);
            return std::nullopt;
        }
        try {
            return it->second.as<T>();
        } catch (const YAML::Exception &e) {
            LOG_ERROR("YAML parsing exception for key '{}': {}", key, e.what());
            return std::nullopt;
        } catch (const std::exception &e) {
            LOG_ERROR("Error fetching value for key '{}': {}", key, e.what());
            return std::nullopt;
        }
    }

    // Specialization for std::string_view
    template<>
    inline std::optional<std::string_view> Configuration::get<std::string_view>(const std::string &key) const {
        auto str_opt = get<std::string>(key);
        if (str_opt) {
            return std::string_view(*str_opt);
        }
        return std::nullopt;
    }

    template<typename T>
    T Configuration::get(const std::string &key, T default_value) const {
        auto value = get<T>(key);
        return value ? *value : default_value;
    }

    // Specialization to force const char* to std::string
    inline std::string Configuration::get(const std::string &key, const char *default_value) const {
        return get<std::string>(key, std::string(default_value));
    }

    inline bool Configuration::contains(const std::string &key) const { return config_map_.contains(key); }

    // Convenience functions for getting configuration values
    template<typename T>
    std::optional<T> get(const std::string &key) {
        return Configuration::getInstance().get<T>(key);
    }

    template<typename T>
    T get(const std::string &key, T default_value) {
        return Configuration::getInstance().get<T>(key, default_value);
    }

    // Specialization to handle const char* as std::string (yaml-cpp misbehaves with const char*)
    inline std::string get(const std::string &key, const char *default_value) {
        return Configuration::getInstance().get(key, default_value);
    }

} // namespace config

#endif // CONFIGURATION_HPP
