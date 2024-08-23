// File: common/formatting/fmt_yaml.hpp

#ifndef FMT_YAML_HPP
#define FMT_YAML_HPP

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

template <>
struct fmt::formatter<YAML::NodeType::value> : formatter<std::string> {
    template <typename FormatContext>
    auto format(YAML::NodeType::value type, FormatContext& ctx) {
        std::string type_str;
        switch (type) {
            case YAML::NodeType::Null: type_str = "Null"; break;
            case YAML::NodeType::Scalar: type_str = "Scalar"; break;
            case YAML::NodeType::Sequence: type_str = "Sequence"; break;
            case YAML::NodeType::Map: type_str = "Map"; break;
            case YAML::NodeType::Undefined: type_str = "Undefined"; break;
        }
        return formatter<std::string>::format(type_str, ctx);
    }
};

#endif // FMT_YAML_HPP
