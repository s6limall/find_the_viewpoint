#ifndef METRICS_COLLECTOR_HPP
#define METRICS_COLLECTOR_HPP

#include <Eigen/Dense>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include "common/logging/logger.hpp"
#include "config/configuration.hpp"
#include "types/viewpoint.hpp"

namespace metrics {

    using json = nlohmann::json;
    using MetricValue = std::variant<int64_t, double, std::string, Eigen::Vector3d, Eigen::Vector3f>;

    class MetricsCollector {
    public:
        static MetricsCollector &getInstance() {
            static MetricsCollector instance;
            return instance;
        }

        void initialize(std::string_view objectName = {}) {
            std::lock_guard lock(mutex);
            if (!objectName.empty())
                currentObject = objectName;
            loadOrCreateDataFile();
        }

        template<typename T>
        void recordMetric(const ViewPoint<T> &viewpoint, std::string_view key, const MetricValue &value) {
            recordMetrics(viewpoint, {{std::string(key), value}});
        }

        template<typename T>
        void recordMetrics(const ViewPoint<T> &viewpoint, const std::unordered_map<std::string, MetricValue> &metrics) {
            recordMetrics(viewpoint.getPosition(), metrics);
        }

        template<typename Derived>
        void recordMetrics(const Eigen::MatrixBase<Derived> &point,
                           const std::unordered_map<std::string, MetricValue> &metrics) {
            try {
                std::lock_guard lock(mutex);
                updateOrAddViewpoint(point, metrics);
                saveDataFile();
            } catch (const std::exception &error) {
                LOG_ERROR("Failed to record metrics: {}", error.what());
                throw;
            }
        }

        void setOutputDirectory(const std::filesystem::path &directory) {
            std::lock_guard lock(mutex);
            outputDirectory = directory;
            loadOrCreateDataFile();
        }

    private:
        MetricsCollector() :
            currentObject(config::get("object.name", "unknown_object")),
            outputDirectory(std::filesystem::current_path()), nextViewpointIndex(0) {
            loadOrCreateDataFile();
        }

        void loadOrCreateDataFile() {
            std::filesystem::create_directories(outputDirectory);
            const auto filePath = getDataFilePath();

            if (std::filesystem::exists(filePath)) {
                std::ifstream file(filePath);
                file >> data;
                nextViewpointIndex = data["viewpoints"].size();
            } else {
                data = {{"viewpoints", json::array()}};
                saveDataFile();
            }
        }

        void saveDataFile() const {
            std::ofstream file(getDataFilePath());
            file << std::setw(4) << data << std::endl;
        }

        std::filesystem::path getDataFilePath() const {
            const auto comparator = config::get("image.comparator.type", "default");
            return outputDirectory / fmt::format("metrics_{}_{}.json", currentObject, comparator);
        }

        template<typename Derived>
        void updateOrAddViewpoint(const Eigen::MatrixBase<Derived> &point,
                                  const std::unordered_map<std::string, MetricValue> &metrics) {
            const std::string viewpointKey = formatPoint(point);
            auto &viewpoints = data["viewpoints"];
            auto existingViewpoint =
                    std::find_if(viewpoints.begin(), viewpoints.end(),
                                 [&viewpointKey](const json &vp) { return vp["key"] == viewpointKey; });

            if (existingViewpoint != viewpoints.end()) {
                updateViewpointData(*existingViewpoint, point, metrics);
            } else {
                viewpoints.push_back(createViewpointData(point, metrics));
                ++nextViewpointIndex;
            }
        }

        template<typename Derived>
        std::string formatPoint(const Eigen::MatrixBase<Derived> &point) const {
            return fmt::format("{:.6f}_{:.6f}_{:.6f}", point[0], point[1], point[2]);
        }

        template<typename Derived>
        json createViewpointData(const Eigen::MatrixBase<Derived> &point,
                                 const std::unordered_map<std::string, MetricValue> &metrics) const {
            json viewpointData = {{"index", nextViewpointIndex},
                                  {"key", formatPoint(point)},
                                  {"timestamp", std::chrono::system_clock::now().time_since_epoch().count()},
                                  {"position", {point[0], point[1], point[2]}}};

            for (const auto &[key, value]: metrics) {
                updateNestedJson(viewpointData, key, value);
            }
            return viewpointData;
        }

        template<typename Derived>
        void updateViewpointData(json &viewpoint, const Eigen::MatrixBase<Derived> &point,
                                 const std::unordered_map<std::string, MetricValue> &metrics) const {
            viewpoint["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
            viewpoint["position"] = {point[0], point[1], point[2]};

            for (const auto &[key, value]: metrics) {
                updateNestedJson(viewpoint, key, value);
            }
        }

        void updateNestedJson(json &j, const std::string &key, const MetricValue &value) const {
            auto keys = splitString(key, '.');
            json *current = &j;

            for (size_t i = 0; i < keys.size() - 1; ++i) {
                if (!current->contains(keys[i])) {
                    (*current)[keys[i]] = json::object();
                }
                current = &(*current)[keys[i]];
            }

            (*current)[keys.back()] = std::visit(
                    [](const auto &v) -> json {
                        using T = std::decay_t<decltype(v)>;
                        if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, double> ||
                                      std::is_same_v<T, std::string>) {
                            return v;
                        } else if constexpr (std::is_same_v<T, Eigen::Vector3d> || std::is_same_v<T, Eigen::Vector3f>) {
                            return {v.x(), v.y(), v.z()};
                        } else {
                            throw std::runtime_error("Unsupported metric type");
                        }
                    },
                    value);
        }

        std::vector<std::string> splitString(const std::string &s, char delimiter) const {
            std::vector<std::string> tokens;
            std::istringstream tokenStream(s);
            std::string token;
            while (std::getline(tokenStream, token, delimiter)) {
                tokens.push_back(token);
            }
            return tokens;
        }

        std::string currentObject;
        std::filesystem::path outputDirectory;
        mutable std::mutex mutex;
        json data;
        size_t nextViewpointIndex;
    };

    template<typename T>
    void recordMetric(const ViewPoint<T> &viewpoint, std::string_view key, const MetricValue &value) {
        MetricsCollector::getInstance().recordMetric(viewpoint, key, value);
    }

    template<typename T>
    void recordMetrics(const ViewPoint<T> &viewpoint, const std::unordered_map<std::string, MetricValue> &metrics) {
        MetricsCollector::getInstance().recordMetrics(viewpoint, metrics);
    }

    template<typename Derived>
    void recordMetrics(const Eigen::MatrixBase<Derived> &point,
                       const std::unordered_map<std::string, MetricValue> &metrics) {
        MetricsCollector::getInstance().recordMetrics(point, metrics);
    }

} // namespace metrics

#endif // METRICS_COLLECTOR_HPP
