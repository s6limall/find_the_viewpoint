// File: cache/viewpoint_cache.hpp

#ifndef VIEWPOINT_CACHE_HPP
#define VIEWPOINT_CACHE_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <optional>
#include <unordered_map>
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
            bool enable_eviction;

            explicit CacheConfig(const size_t max_size = 10000, T score_threshold = 0.8, T base_radius = 0.1,
                                 const bool enable_eviction = true) :
                max_size(config::get("cache.max_size", max_size)),
                score_threshold(config::get("cache.score_threshold", score_threshold)),
                base_radius(config::get("cache.base_radius", base_radius)),
                enable_eviction(config::get("cache.evict", enable_eviction)) {}
        };


        explicit ViewpointCache(CacheConfig config = CacheConfig{}) : config_(std::move(config)) {
            LOG_INFO("ViewpointCache initialized with max_size: {}, score_threshold: {}, base_radius: {}, eviction: {}",
                     config_.max_size, config_.score_threshold, config_.base_radius,
                     config_.enable_eviction ? "enabled" : "disabled");
        }


        // Nearest neighbor search using spatial hashing (https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
        std::optional<T> query(const Eigen::Vector3<T> &position) {
            auto hash = computeSpatialHash(position);
            auto it = spatial_index_.find(hash);
            if (it == spatial_index_.end()) {
                LOG_TRACE("Cache miss for position: {}", position);
                return std::nullopt;
            }

            auto &entry = *it->second;
            T distance = (position - entry.viewpoint.getPosition()).norm();

            if (shouldUseCachedValue(entry.viewpoint.getScore(), distance, entry.radius)) {
                updateLRU(it->second);
                LOG_DEBUG("Cache hit for position: {}, score: {}", position, entry.viewpoint.getScore());
                return entry.viewpoint.getScore();
            }

            LOG_TRACE("Cache miss for position: {}", position);
            return std::nullopt;
        }


        // LRU cache insertion with optional eviction - LRU policy
        void update(const ViewPoint<T> &viewpoint) {
            auto hash = computeSpatialHash(viewpoint.getPosition());
            auto it = spatial_index_.find(hash);
            if (it != spatial_index_.end()) {
                auto &entry = *it->second;
                T distance = (viewpoint.getPosition() - entry.viewpoint.getPosition()).norm();
                if (distance <= entry.radius) {
                    if (viewpoint.getScore() > entry.viewpoint.getScore()) {
                        entry.viewpoint = viewpoint;
                        entry.radius = computeAdaptiveRadius(viewpoint.getScore());
                        updateLRU(it->second);
                        LOG_DEBUG("Updated cache entry for position: {}, new score: {}, new radius: {}",
                                  viewpoint.getPosition(), viewpoint.getScore(), entry.radius);
                    }
                } else {
                    insert(viewpoint);
                }
            } else {
                insert(viewpoint);
            }
        }

        // Modify the insert method to handle potential updates
        void insert(const ViewPoint<T> &viewpoint) {
            auto hash = computeSpatialHash(viewpoint.getPosition());
            auto it = spatial_index_.find(hash);
            if (it != spatial_index_.end()) {
                auto &entry = *it->second;
                T distance = (viewpoint.getPosition() - entry.viewpoint.getPosition()).norm();
                if (distance <= entry.radius) {
                    if (viewpoint.getScore() > entry.viewpoint.getScore()) {
                        entry.viewpoint = viewpoint;
                        entry.radius = computeAdaptiveRadius(viewpoint.getScore());
                        updateLRU(it->second);
                        LOG_DEBUG("Updated existing cache entry for position: {}, new score: {}, new radius: {}",
                                  viewpoint.getPosition(), viewpoint.getScore(), entry.radius);
                    }
                    return;
                }
            }

            if (config_.enable_eviction && lru_list_.size() >= config_.max_size) {
                evict();
            } else if (!config_.enable_eviction && lru_list_.size() >= config_.max_size) {
                LOG_WARN("Cache full and eviction disabled. Skipping insertion of viewpoint: {}, score: {}",
                         viewpoint.getPosition(), viewpoint.getScore());
                return;
            }

            T radius = computeAdaptiveRadius(viewpoint.getScore());

            lru_list_.emplace_front(CacheEntry{viewpoint, radius, std::chrono::steady_clock::now()});
            spatial_index_[hash] = lru_list_.begin();

            LOG_DEBUG("Inserted new cache entry for position: {}, score: {}, radius: {}", viewpoint.getPosition(),
                      viewpoint.getScore(), radius);
        }


        void clear() {
            lru_list_.clear();
            spatial_index_.clear();
            LOG_INFO("ViewpointCache cleared");
        }

        [[nodiscard]] size_t size() const noexcept { return lru_list_.size(); }
        [[nodiscard]] bool empty() const noexcept { return lru_list_.empty(); }

    private:
        CacheConfig config_;
        std::list<CacheEntry> lru_list_;
        std::unordered_map<size_t, typename std::list<CacheEntry>::iterator> spatial_index_;

        // Adaptive radius calculation (higher the score, less the likelihood to use cache)
        // For low scoring viewpoints, we can use nearby cached points as proxies instead of re-evaluating
        T computeAdaptiveRadius(T score) const noexcept {
            return config_.base_radius * (1 - std::pow(score / config_.score_threshold, 2));
        }

        // Decide whether to use the cached value (distance/radius check)
        bool shouldUseCachedValue(T score, T distance, T radius) const noexcept {
            T normalized_distance = distance / radius;
            T score_factor = score / config_.score_threshold;
            return normalized_distance < (1 - score_factor);
        }


        void evict() {
            if (!config_.enable_eviction || lru_list_.empty())
                return;

            auto oldest_it = std::prev(lru_list_.end());
            spatial_index_.erase(computeSpatialHash(oldest_it->viewpoint.getPosition()));
            lru_list_.pop_back();

            LOG_DEBUG("Evicted oldest entry from cache");
        }


        // LRU update
        void updateLRU(typename std::list<CacheEntry>::iterator it) {
            if (!config_.enable_eviction)
                return;

            if (it != lru_list_.begin()) {
                lru_list_.splice(lru_list_.begin(), lru_list_, it);
                spatial_index_[computeSpatialHash(it->viewpoint.getPosition())] = lru_list_.begin();
            }
            it->last_access = std::chrono::steady_clock::now();
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
