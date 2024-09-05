// File: config/configuration.hpp

#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "common/logging/logger.hpp"

/*template<>
struct YAML::convert<std::basic_string_view<char>> {
    static Node encode(const std::basic_string_view<char> &rhs) {
        return Node(std::string(rhs)); // Convert string_view to string
    }

    static bool decode(const Node &node, std::basic_string_view<char> &rhs) {
        if (!node.IsScalar()) {
            return false;
        }
        static auto value = node.as<std::string>();
        rhs = std::basic_string_view<char>(value);
        return true;
    }
};*/

namespace config {
    class Configuration {
    public:
        using ChangeCallback = std::function<void(const std::string &, const YAML::Node &)>;

        Configuration(const Configuration &) = delete;

        Configuration &operator=(const Configuration &) = delete;

        Configuration(Configuration &&) = delete;

        Configuration &operator=(Configuration &&) = delete;

        ~Configuration() = default;

        explicit Configuration(std::string filename);

        // Initialize the configuration with a custom filepath
        static void initialize(const std::string &filename);

        // Get the singleton instance with optional filename
        static Configuration &getInstance(const std::string &filename = "");

        // Check if a key exists in the configuration
        [[nodiscard]] bool contains(const std::string &key) const;

        // Helper function to log entire configuration
        void show() const;

        // Get a value of type T from the configuration
        template<typename T>
        [[nodiscard]] std::optional<T> get(const std::string &key) const;

        // Get a value of type T from the configuration with a default value
        template<typename T>
        [[nodiscard]] T get(const std::string &key, T default_value) const;

        // Specialization to handle const char* as std::string
        [[nodiscard]] std::string get(const std::string &key, const char *default_value) const;

        // New functionality: Set a value in the configuration
        template<typename T>
        bool set(const std::string &key, const T &value);

        // New functionality: Reload the configuration from file
        void reload();

        // New functionality: Register a callback for configuration changes
        void registerChangeCallback(ChangeCallback callback);

    private:
        std::unordered_map<std::string, YAML::Node> config_map_;
        static constexpr std::string_view default_filename_ = "configuration.yaml";
        static std::shared_ptr<Configuration> instance_;
        static std::once_flag init_flag_;
        std::string filename_;
        mutable std::shared_mutex mutex_;
        std::vector<ChangeCallback> change_callbacks_;

        // Load the entire configuration into a map
        void load(const YAML::Node &node, const std::string &prefix = "");

        // Helper function to split keys
        static std::vector<std::string> split(const std::string &str, char delimiter);

        // Helper function to notify change callbacks
        void notifyChangeCallbacks(const std::string &key, const YAML::Node &value) const;
    };

    // Template definitions
    template<typename T>
    std::optional<T> Configuration::get(const std::string &key) const {
        std::shared_lock lock(mutex_);
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

    template<typename T>
    T Configuration::get(const std::string &key, T default_value) const {
        auto value = get<T>(key);
        return value ? *value : default_value;
    }

    // Specialization to force const char* to std::string
    inline std::string Configuration::get(const std::string &key, const char *default_value) const {
        return get<std::string>(key, std::string(default_value));
    }

    template<typename T>
    bool Configuration::set(const std::string &key, const T &value) {
        std::unique_lock lock(mutex_);
        try {
            YAML::Node node;
            node = value;
            config_map_[key] = node;
            notifyChangeCallbacks(key, node);
            return true;
        } catch (const std::exception &e) {
            LOG_ERROR("Error setting value for key '{}': {}", key, e.what());
            return false;
        }
    }

    inline void initialize(const std::string &filename = {}) { Configuration::initialize(filename); }

    inline bool Configuration::contains(const std::string &key) const {
        std::shared_lock lock(mutex_);
        return config_map_.contains(key);
    }

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

    // New convenience functions
    template<typename T>
    bool set(const std::string &key, const T &value) {
        return Configuration::getInstance().set<T>(key, value);
    }

    inline bool contains(const std::string &key) { return Configuration::getInstance().contains(key); }

    inline void reload() { Configuration::getInstance().reload(); }

    inline void show() { Configuration::getInstance().show(); }

    inline void registerChangeCallback(Configuration::ChangeCallback callback) {
        Configuration::getInstance().registerChangeCallback(std::move(callback));
    }
} // namespace config

#endif // CONFIGURATION_HPP
