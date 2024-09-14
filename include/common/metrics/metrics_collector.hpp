#ifndef METRICS_COLLECTOR_HPP
#define METRICS_COLLECTOR_HPP

#include <Eigen/Dense>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string_view>
#include <unordered_map>
#include <variant>

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
            deleteExistingFile();
            resetData();
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
            deleteExistingFile();
            resetData();
        }

    private:
        MetricsCollector() :
            currentObject(config::get("object.name", "unknown_object")),
            outputDirectory(std::filesystem::current_path()), nextViewpointIndex(0) {
            deleteExistingFile();
            resetData();
        }

        void deleteExistingFile() {
            const auto filePath = getDataFilePath();
            if (std::filesystem::exists(filePath)) {
                std::filesystem::remove(filePath);
                LOG_INFO("Existing metrics file deleted: {}", filePath.string());
            }
        }

        void resetData() {
            data = {{"viewpoints", json::array()}};
            nextViewpointIndex = 0;
        }

        void saveDataFile() const {
            const auto filePath = getDataFilePath();
            std::filesystem::create_directories(filePath.parent_path());
            std::ofstream file(filePath);
            if (!file) {
                throw std::runtime_error("Failed to create file for writing: " + filePath.string());
            }
            file << std::setw(4) << data << std::endl;
            LOG_INFO("Metrics saved to file: {}", filePath.string());
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

            for (auto it = keys.begin(); it != keys.end() - 1; ++it) {
                if (!current->contains(*it)) {
                    (*current)[*it] = json::object();
                }
                current = &(*current)[*it];
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

    // Free functions for convenience
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
