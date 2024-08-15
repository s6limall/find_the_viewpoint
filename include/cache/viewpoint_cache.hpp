// File: cache/viewpoint_cache.hpp

#ifndef VIEWPOINT_CACHE_HPP
#define VIEWPOINT_CACHE_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>
#include "common/logging/logger.hpp"
#include "types/viewpoint.hpp"

namespace cache {

    template<typename T = double>
    class ViewpointCache {
    public:
        struct CacheEntry {
            ViewPoint<T> viewpoint;
            T radius;
            std::chrono::steady_clock::time_point last_access;
        };

        struct CacheConfig {
            size_t max_size;
            T score_threshold;
            T base_radius;
            bool evict;

            explicit CacheConfig(const size_t max_size = 10000, T score_threshold = 0.8, T base_radius = 0.1,
                                 const bool evict = true) :
                max_size(config::get("cache.max_size", max_size)),
                score_threshold(config::get("cache.score_threshold", score_threshold)),
                base_radius(config::get("cache.base_radius", base_radius)), evict(config::get("cache.evict", evict)) {}
        };

        explicit ViewpointCache(CacheConfig config = CacheConfig{}) : config_(std::move(config)) {
            LOG_INFO(
                    "Viewpoint Cache initialized with max_size: {}, score_threshold: {}, base_radius: {}, eviction: {}",
                    config_.max_size, config_.score_threshold, config_.base_radius,
                    config_.evict ? "enabled" : "disabled");
        }

        std::optional<T> query(const Eigen::Vector3<T> &position) const noexcept {
            size_t hash = computeSpatialHash(position);
            auto it = spatial_index_.find(hash);
            if (it == spatial_index_.end()) {
                return std::nullopt;
            }

            const auto &bucket = it->second;
            auto entry_it = std::find_if(bucket.begin(), bucket.end(), [&](const auto &entry) {
                return shouldUseCachedValue(entry.viewpoint.getScore(),
                                            (position - entry.viewpoint.getPosition()).norm(), entry.radius);
            });

            if (entry_it != bucket.end()) {
                LOG_DEBUG("Cache hit for position: {}, score: {}", position, entry_it->viewpoint.getScore());
                return entry_it->viewpoint.getScore();
            }

            LOG_TRACE("Cache miss for position: {}", position);
            return std::nullopt;
        }

        void update(const ViewPoint<T> &viewpoint) {
            size_t hash = computeSpatialHash(viewpoint.getPosition());
            auto &bucket = spatial_index_[hash];

            auto it = std::find_if(bucket.begin(), bucket.end(), [&](const auto &entry) {
                return (viewpoint.getPosition() - entry.viewpoint.getPosition()).norm() <= entry.radius;
            });

            if (it != bucket.end()) {
                if (viewpoint.getScore() > it->viewpoint.getScore()) {
                    it->viewpoint = viewpoint;
                    it->radius = computeAdaptiveRadius(viewpoint.getScore());
                    it->last_access = std::chrono::steady_clock::now();
                    LOG_DEBUG("Updated cache entry for position: {}, new score: {}, new radius: {}",
                              viewpoint.getPosition(), viewpoint.getScore(), it->radius);
                }
            } else {
                insert(viewpoint);
            }
        }

        void insert(const ViewPoint<T> &viewpoint) {
            if (size() >= config_.max_size) {
                if (config_.evict) {
                    evict();
                } else {
                    LOG_WARN("Cache full and eviction disabled. Skipping insertion of viewpoint: {}, score: {}",
                             viewpoint.getPosition(), viewpoint.getScore());
                    return;
                }
            }

            T radius = computeAdaptiveRadius(viewpoint.getScore());
            size_t hash = computeSpatialHash(viewpoint.getPosition());
            spatial_index_[hash].emplace_back(CacheEntry{viewpoint, radius, std::chrono::steady_clock::now()});

            LOG_DEBUG("Inserted new cache entry for position: {}, score: {}, radius: {}", viewpoint.getPosition(),
                      viewpoint.getScore(), radius);
        }

        void clear() noexcept {
            spatial_index_.clear();
            LOG_INFO("ViewpointCache cleared");
        }

        [[nodiscard]] size_t size() const noexcept {
            size_t total_size = 0;
            for (const auto &bucket: spatial_index_) {
                total_size += bucket.second.size();
            }
            return total_size;
        }

        [[nodiscard]] bool empty() const noexcept { return spatial_index_.empty(); }

    private:
        CacheConfig config_;
        std::unordered_map<size_t, std::vector<CacheEntry>> spatial_index_;

        [[nodiscard]] T computeAdaptiveRadius(T score) const noexcept {
            return config_.base_radius * (1 - std::pow(score / config_.score_threshold, 2));
        }

        bool shouldUseCachedValue(T score, T distance, T radius) const noexcept {
            T normalized_distance = distance / radius;
            T score_factor = score / config_.score_threshold;
            return normalized_distance < (1 - score_factor);
        }

        void evict() noexcept {
            if (spatial_index_.empty())
                return;

            auto oldest_bucket_it =
                    std::min_element(spatial_index_.begin(), spatial_index_.end(), [](const auto &a, const auto &b) {
                        return !a.second.empty() &&
                               (b.second.empty() || a.second.front().last_access < b.second.front().last_access);
                    });

            if (oldest_bucket_it != spatial_index_.end() && !oldest_bucket_it->second.empty()) {
                oldest_bucket_it->second.erase(
                        std::min_element(oldest_bucket_it->second.begin(), oldest_bucket_it->second.end(),
                                         [](const auto &a, const auto &b) { return a.last_access < b.last_access; }));

                if (oldest_bucket_it->second.empty()) {
                    spatial_index_.erase(oldest_bucket_it);
                }

                LOG_DEBUG("Evicted oldest entry from cache");
            }
        }

        // Simplified spatial locality hashing function using Z-order curve
        // https://en.wikipedia.org/wiki/Z-order_curve
        size_t computeSpatialHash(const Eigen::Vector3<T> &position) const noexcept {
            const T cell_size = config_.base_radius;

            const int x = static_cast<int>(std::floor(position[0] / cell_size));
            const int y = static_cast<int>(std::floor(position[1] / cell_size));
            const int z = static_cast<int>(std::floor(position[2] / cell_size));

            size_t hash = 0;
            for (size_t i = 0; i < sizeof(int) * 8; ++i) {
                hash |= ((x & (1 << i)) << (2 * i)) | ((y & (1 << i)) << (2 * i + 1)) | ((z & (1 << i)) << (2 * i + 2));
            }

            return hash % config_.max_size;
        }
    };

} // namespace cache

#endif // VIEWPOINT_CACHE_HPP
