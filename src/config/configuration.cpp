// File: config/configuration.cpp

#include "config/configuration.hpp"

namespace config {

    std::shared_ptr<Configuration> Configuration::instance_;
    std::once_flag Configuration::init_flag_;

    Configuration &Configuration::getInstance(const std::string &filename) {
        std::call_once(init_flag_, [&filename] {
            instance_ = std::make_shared<Configuration>(filename.empty() ? default_filename_.data() : filename);
        });
        return *instance_;
    }

    Configuration::Configuration(const std::string &filename) {
        LOG_INFO("Loading configuration from file: {}", filename);

        try {
            const YAML::Node root = YAML::LoadFile(filename);
            LOG_INFO("Configuration file '{}' loaded successfully.", filename);
            load(root);
        } catch (const YAML::Exception &e) {
            LOG_CRITICAL("YAML exception while loading configuration: {}", e.what());
            throw std::runtime_error("YAML exception");
        } catch (const std::exception &e) {
            LOG_CRITICAL("Unknown error loading configuration: {}", e.what());
            throw std::runtime_error("Unknown error loading configuration");
        }
    }

    void Configuration::load(const YAML::Node &node, const std::string &prefix) {
    for (const auto &it: node) {
        std::string key = prefix.empty() ? it.first.as<std::string>() : prefix + "." + it.first.as<std::string>();
        if (it.second.IsMap()) {
            LOG_DEBUG("Loading nested map for key: '{}'", key);
            load(it.second, key); // Recursively load nested maps
        } else {
            config_map_[key] = it.second;
            LOG_DEBUG("Loaded key: '{}', value type: '{}', value: '{}'",
                      key,
                      it.second.Type(),
                      it.second.IsScalar() ? it.second.as<std::string>() : "[non-scalar]");
        }
    }
}


    std::vector<std::string> Configuration::split(const std::string &str, const char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(str);
        while (std::getline(tokenStream, token, delimiter)) {
            if (!token.empty()) {
                tokens.push_back(std::move(token));
            }
        }
        return tokens;
    }

    void Configuration::show() const {
        LOG_INFO("Configuration details:");
        for (const auto &entry: config_map_) {
            try {
                LOG_INFO("{}: {}", entry.first, entry.second.as<std::string>());
            } catch (const YAML::Exception &e) {
                LOG_ERROR("Error logging configuration key '{}': {}", entry.first, e.what());
            }
        }
    }
}
