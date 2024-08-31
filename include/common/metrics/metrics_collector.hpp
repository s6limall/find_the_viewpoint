// File: common/metrics/metrics_collector.hpp

#ifndef METRICS_COLLECTOR_HPP
#define METRICS_COLLECTOR_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include "common/logging/logger.hpp"
#include "config/configuration.hpp"

namespace metrics {

    using MetricVariant = std::variant<int64_t, double, std::string, Eigen::Vector3d, Eigen::Vector3f>;

    class MetricsCollector {
    public:
        struct MetricError final : std::runtime_error {
            using std::runtime_error::runtime_error;
        };

        static MetricsCollector &getInstance() {
            static MetricsCollector instance;
            return instance;
        }

        void initialize(const std::optional<std::string_view> &custom_object_name = std::nullopt) {
            std::lock_guard lock(mutex_);
            if (custom_object_name) {
                current_object_ = *custom_object_name;
            }
            init();
        }

        template<typename PointType>
        void recordMetric(const PointType &point, std::string_view metric_name, const MetricVariant &value) {
            try {
                std::call_once(init_flag_, [this] { init(); });
                storeMetric(point, metric_name, value);
            } catch (const std::exception &e) {
                LOG_ERROR("Failed to record metric: {}", e.what());
                throw MetricError(fmt::format("Failed to record metric '{}': {}", metric_name, e.what()));
            }
        }

        template<typename PointType, typename MetricsContainer>
        void recordMetrics(const PointType &point, const MetricsContainer &metrics) {
            try {
                std::call_once(init_flag_, [this] { init(); });
                for (const auto &[metric_name, value]: metrics) {
                    storeMetric(point, metric_name, value);
                }
            } catch (const std::exception &e) {
                LOG_ERROR("Failed to record metrics: {}", e.what());
                throw MetricError(fmt::format("Failed to record metrics: {}", e.what()));
            }
        }

        std::future<void> flushAsync() const {
            return std::async(std::launch::async, [this] { saveMetrics(); });
        }

        void flush() const { saveMetrics(); }

        std::vector<std::pair<std::string, MetricVariant>> getMetrics() const {
            std::lock_guard lock(mutex_);
            std::vector<std::pair<std::string, MetricVariant>> result;
            for (const auto &[key, entries]: data_) {
                for (const auto &entry: entries) {
                    for (const auto &value: entry.values) {
                        result.emplace_back(entry.name, value.value);
                    }
                }
            }
            return result;
        }

        void clear() {
            std::lock_guard lock(mutex_);
            data_.clear();
            row_index_ = 0;
        }

        void setOutputDirectory(const std::filesystem::path &dir) {
            std::lock_guard lock(mutex_);
            output_directory_ = dir;
        }

    private:
        MetricsCollector() : row_index_(0) {}
        ~MetricsCollector() {
            try {
                flush();
            } catch (const std::exception &e) {
                LOG_ERROR("Failed to flush metrics on destruction: {}", e.what());
            }
        }

        void init() {
            if (current_object_.empty()) {
                current_object_ = config::get<std::string>("object.name", "unknown_object");
            }
            start_time_ = std::chrono::steady_clock::now();
            data_.clear();
            row_index_ = 0;
            LOG_INFO("Metrics collection initialized for object: {}", current_object_);
        }

        template<typename PointType>
        void storeMetric(const PointType &point, std::string_view metric_name, const MetricVariant &value) {
            const auto current_time = std::chrono::steady_clock::now();
            const auto duration =
                    std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time_).count();

            std::lock_guard lock(mutex_);
            auto &entries = data_[getPointKey(point)];
            auto it = std::find_if(entries.begin(), entries.end(),
                                   [&](const auto &entry) { return entry.name == metric_name; });
            if (it == entries.end()) {
                entries.emplace_back(MetricEntry{std::string(metric_name), {}});
                it = std::prev(entries.end());
            }

            it->values.emplace_back(MetricValue{duration, value, row_index_++});
        }

        template<typename PointType>
        std::string getPointKey(const PointType &point) const {
            if constexpr (requires { point.getPosition(); }) {
                const auto &pos = point.getPosition();
                return fmt::format("{:.6f}_{:.6f}_{:.6f}", pos.x(), pos.y(), pos.z());
            } else if constexpr (Eigen::MatrixBase<PointType>::RowsAtCompileTime == 3 ||
                                 Eigen::MatrixBase<PointType>::ColsAtCompileTime == 1) {
                return fmt::format("{:.6f}_{:.6f}_{:.6f}", point(0), point(1), point(2));
            } else if constexpr (std::is_same_v<PointType, Eigen::VectorXd>) {
                if (point.size() == 3) {
                    return fmt::format("{:.6f}_{:.6f}_{:.6f}", point(0), point(1), point(2));
                } else {
                    throw std::invalid_argument("Vector size must be 3 for getPointKey.");
                }
            } else {
                throw std::invalid_argument("Unsupported point type for getPointKey.");
            }
        }

        static std::string formatValue(const MetricVariant &value) {
            return std::visit(
                    []<typename T0>(T0 &&arg) -> std::string {
                        using T = std::decay_t<T0>;
                        if constexpr (std::is_same_v<T, Eigen::Vector3d> || std::is_same_v<T, Eigen::Vector3f>) {
                            return fmt::format("{:.6f},{:.6f},{:.6f}", arg.x(), arg.y(), arg.z());
                        } else if constexpr (std::is_arithmetic_v<T>) {
                            return fmt::format("{}", arg);
                        } else if constexpr (std::is_same_v<T, std::string>) {
                            return arg;
                        } else {
                            return "Unsupported Type";
                        }
                    },
                    value);
        }

        void saveMetrics() const {
            std::lock_guard lock(mutex_);
            const auto comparator = config::get("image.comparator.type", "NA");
            const auto filename = output_directory_ / fmt::format("metrics_{}_{}.csv", current_object_, comparator);

            std::filesystem::create_directories(output_directory_);

            std::ofstream file(filename);
            if (!file.is_open()) {
                LOG_ERROR("Failed to open file for writing metrics: {}", filename.string());
                throw MetricError(fmt::format("Failed to open file for writing metrics: {}", filename.string()));
            }

            // Collect all unique metric names (column headers)
            std::set<std::string> all_metric_names;
            for (const auto &[_, entries]: data_) {
                for (const auto &entry: entries) {
                    all_metric_names.insert(entry.name);
                }
            }

            // Write CSV header with all metric names
            file << "timestamp,point_id";
            for (const auto &metric_name: all_metric_names) {
                file << "," << metric_name;
            }
            file << '\n';

            // Collect all data points
            struct RowData {
                int64_t timestamp;
                std::string point_id;
                std::unordered_map<std::string, std::string> metrics;
            };
            std::vector<RowData> all_rows;

            for (const auto &[key, entries]: data_) {
                for (const auto &entry: entries) {
                    for (const auto &value: entry.values) {
                        auto it = std::find_if(all_rows.begin(), all_rows.end(), [&](const RowData &row) {
                            return row.timestamp == value.timestamp && row.point_id == key;
                        });

                        if (it == all_rows.end()) {
                            all_rows.push_back({value.timestamp, key, {}});
                            it = std::prev(all_rows.end());
                        }

                        it->metrics[entry.name] = formatValue(value.value);
                    }
                }
            }

            // Sort rows by timestamp
            std::ranges::sort(all_rows, [](const RowData &a, const RowData &b) { return a.timestamp < b.timestamp; });

            // Write sorted data
            for (const auto &row: all_rows) {
                file << row.timestamp << "," << row.point_id;
                for (const auto &metric_name: all_metric_names) {
                    file << ",";
                    auto it = row.metrics.find(metric_name);
                    if (it != row.metrics.end()) {
                        file << it->second;
                    }
                }
                file << '\n';
            }

            LOG_INFO("Metrics saved to file: {}", filename.string());
        }


        struct MetricValue {
            int64_t timestamp;
            MetricVariant value;
            size_t row_index;
        };

        struct MetricEntry {
            std::string name;
            std::vector<MetricValue> values;
        };

        std::unordered_map<std::string, std::vector<MetricEntry>> data_;
        std::string current_object_;
        std::chrono::steady_clock::time_point start_time_;
        mutable std::mutex mutex_;
        mutable std::once_flag init_flag_;
        size_t row_index_;
        std::filesystem::path output_directory_ = std::filesystem::current_path();
    };

    template<typename PointType>
    void recordMetric(const PointType &point, std::string_view metric_name, const MetricVariant &value) {
        MetricsCollector::getInstance().recordMetric(point, metric_name, value);
    }

    template<typename PointType>
    void recordMetrics(const PointType &point, std::span<const std::pair<std::string_view, MetricVariant>> metrics) {
        MetricsCollector::getInstance().recordMetrics(point, metrics);
    }

    template<typename PointType>
    void recordMetrics(const PointType &point,
                       std::initializer_list<std::pair<std::string_view, MetricVariant>> metrics) {
        MetricsCollector::getInstance().recordMetrics(point, metrics);
    }

} // namespace metrics

#endif // METRICS_COLLECTOR_HPP
