// File: config/configuration.cpp

#include "config/configuration.hpp"


namespace config {

    Configuration &Configuration::getInstance(const std::string &filename) {
        static Configuration instance(filename.empty() ? default_filename_ : filename);
        return instance;
    }


    Configuration::Configuration(const std::string &filename) {
        spdlog::info("Loading configuration from file: {}", filename);

        try {
            YAML::Node root = YAML::LoadFile(filename);
            spdlog::info("Configuration file '{}' loaded successfully.", filename);
            load(root);
            logConfiguration();
        } catch (const YAML::Exception &e) {
            spdlog::critical("YAML exception while loading configuration: {}", e.what());
            throw std::runtime_error("YAML exception");
        } catch (const std::exception &e) {
            spdlog::critical("Unknown error loading configuration: {}", e.what());
            throw std::runtime_error("Unknown error loading configuration");
        }
    }

    void Configuration::load(const YAML::Node &node, const std::string &prefix) {
        for (const auto &it: node) {
            std::string key = prefix.empty() ? it.first.as<std::string>() : prefix + "." + it.first.as<std::string>();
            if (it.second.IsMap()) {
                load(it.second, key); // Recursively load nested maps
            } else {
                config_map_[key] = it.second;
                spdlog::debug("Loaded key: '{}', value: '{}'", key, it.second.as<std::string>());
            }
        }
    }


    std::vector<std::string> Configuration::split(const std::string &str, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(str);
        while (std::getline(tokenStream, token, delimiter)) {
            if (!token.empty()) {
                tokens.push_back(token);
            }
        }
        return tokens;
    }

    void Configuration::logConfiguration() const {
        spdlog::info("Configuration details:");
        for (const auto &entry: config_map_) {
            try {
                spdlog::info("{}: {}", entry.first, entry.second.as<std::string>());
            } catch (const YAML::Exception &e) {
                spdlog::error("Error logging configuration key '{}': {}", entry.first, e.what());
            }
        }
    }


}
