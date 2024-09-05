// File: common/state/state.hpp

#ifndef STATE_HPP
#define STATE_HPP

#include <any>
#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace state {

    class State {
    public:
        using StateValue = std::any;
        using StateCallback = std::function<void(std::string_view, const StateValue &)>;

        static State &instance() noexcept {
            static State instance;
            return instance;
        }

        template<typename T>
        State &set(std::string_view key, T &&value) {
            {
                std::unique_lock state_lock(state_mutex_);
                state_.emplace(std::piecewise_construct, std::forward_as_tuple(key),
                               std::forward_as_tuple(std::forward<T>(value)))
                        .first->second = std::forward<T>(value);
            }
            notifyObservers(key);
            return *this;
        }

        template<typename T>
        [[nodiscard]] T get(const std::string_view key, const T &default_value = T{}) const {
            std::shared_lock state_lock(state_mutex_);
            const auto it = state_.find(std::string(key));
            if (it != state_.end()) {
                if (const auto *value = std::any_cast<T>(&it->second)) {
                    return *value;
                }
                throw std::runtime_error("State type mismatch for key: " + std::string(key));
            }
            return default_value;
        }

        template<typename T>
        [[nodiscard]] std::optional<T> getOptional(const std::string_view key) const noexcept {
            std::shared_lock state_lock(state_mutex_);
            const auto it = state_.find(std::string(key));
            if (it != state_.end()) {
                if (const auto *value = std::any_cast<T>(&it->second)) {
                    return *value;
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] bool contains(const std::string_view key) const noexcept {
            std::shared_lock state_lock(state_mutex_);
            return state_.contains(std::string(key));
        }

        State &remove(const std::string_view key) noexcept {
            {
                std::unique_lock state_lock(state_mutex_);
                state_.erase(std::string(key));
            }
            notifyObservers(key);
            return *this;
        }

        void registerCallback(const std::string_view key, StateCallback callback) {
            std::unique_lock observers_lock(observers_mutex_);
            observers_[std::string(key)].emplace_back(std::move(callback));
        }

        // Static convenience functions
        template<typename T>
        static State &setState(std::string_view key, T &&value) {
            return instance().set(key, std::forward<T>(value));
        }

        template<typename T>
        [[nodiscard]] static T getState(const std::string_view key, const T &default_value = T{}) {
            return instance().get<T>(key, default_value);
        }

        template<typename T>
        [[nodiscard]] static std::optional<T> getOptionalState(const std::string_view key) noexcept {
            return instance().getOptional<T>(key);
        }

        [[nodiscard]] static bool containsState(const std::string_view key) noexcept {
            return instance().contains(key);
        }

        static State &removeState(const std::string_view key) noexcept { return instance().remove(key); }

        static void registerStateCallback(const std::string_view key, StateCallback callback) {
            instance().registerCallback(key, std::move(callback));
        }

        State(const State &) = delete;
        State &operator=(const State &) = delete;
        State(State &&) = delete;
        State &operator=(State &&) = delete;

    private:
        State() = default;

        mutable std::shared_mutex state_mutex_;
        std::unordered_map<std::string, StateValue> state_;

        mutable std::shared_mutex observers_mutex_;
        std::unordered_map<std::string, std::vector<StateCallback>> observers_;

        void notifyObservers(const std::string_view key) const noexcept {
            std::vector<StateCallback> callbacks;
            {
                std::shared_lock observers_lock(observers_mutex_);
                const auto it = observers_.find(std::string(key));
                if (it != observers_.end()) {
                    callbacks = it->second;
                }
            }
            for (const auto &callback: callbacks) {
                if (contains(key)) {
                    std::shared_lock state_lock(state_mutex_);
                    callback(key, state_.at(std::string(key)));
                } else {
                    callback(key, StateValue{});
                }
            }
        }
    };

    // Convenience functions

    template<typename T>
    void set(std::string_view key, T &&value) {
        State::instance().set(key, std::forward<T>(value));
    }

    template<typename T>
    [[nodiscard]] auto get(const std::string_view key, T &&default_value) {
        return State::instance().get<std::decay_t<T>>(key, std::forward<T>(default_value));
    }

    template<typename T>
    [[nodiscard]] std::optional<std::decay_t<T>> getOptional(const std::string_view key) noexcept {
        return State::instance().getOptional<std::decay_t<T>>(key);
    }

    [[nodiscard]] inline bool contains(const std::string_view key) noexcept { return State::instance().contains(key); }

    inline void remove(const std::string_view key) noexcept { State::instance().remove(key); }

    inline void registerCallback(const std::string_view key, State::StateCallback callback) {
        State::instance().registerCallback(key, std::move(callback));
    }

} // namespace state

#endif // STATE_HPP
