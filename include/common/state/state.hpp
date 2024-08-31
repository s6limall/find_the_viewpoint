// File: common/state/state.hpp

#ifndef STATE_HPP
#define STATE_HPP

#include <functional>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

namespace state {

    class State {
    public:
        using StateValue = std::variant<int, double, std::string, bool>;
        using StateCallback = std::function<void(std::string_view, const StateValue &)>;

        static State &instance() noexcept {
            static State instance;
            return instance;
        }

        template<typename T>
        State &set(const std::string_view key, T &&value) {
            if constexpr (std::is_convertible_v<T, std::string_view>) {
                std::unique_lock lock(mutex_);
                state_[std::string(key)] = std::string(value);
            } else {
                static_assert(is_state_value_type_v<std::decay_t<T>>, "Unsupported type for StateValue");
                std::unique_lock lock(mutex_);
                state_[std::string(key)] = std::forward<T>(value);
            }
            notifyObservers(key, state_.at(std::string(key)));
            return *this;
        }


        template<typename T>
        [[nodiscard]] T get(const std::string_view key, const T &default_value = T{}) const {
            static_assert(is_state_value_type_v<T>, "Unsupported type for StateValue");
            std::shared_lock lock(mutex_);
            if (const auto it = state_.find(std::string(key)); it != state_.end()) {
                if (const auto *value = std::get_if<T>(&it->second)) {
                    return *value;
                }
                throw std::runtime_error("State type mismatch for key: " + std::string(key));
            }
            return default_value;
        }

        template<typename T>
        [[nodiscard]] std::optional<T> getOptional(const std::string_view key) const noexcept {
            static_assert(is_state_value_type_v<T>, "Unsupported type for StateValue");
            std::shared_lock lock(mutex_);
            if (const auto it = state_.find(std::string(key)); it != state_.end()) {
                if (const auto *value = std::get_if<T>(&it->second)) {
                    return *value;
                }
            }
            return std::nullopt;
        }

        [[nodiscard]] bool contains(const std::string_view key) const noexcept {
            std::shared_lock lock(mutex_);
            return state_.contains(std::string(key));
        }

        State &remove(const std::string_view key) noexcept {
            {
                std::unique_lock lock(mutex_);
                state_.erase(std::string(key));
            }
            notifyObservers(key, std::nullopt);
            return *this;
        }

        void registerCallback(const std::string_view key, StateCallback callback) {
            std::unique_lock lock(mutex_);
            observers_[std::string(key)].push_back(std::move(callback));
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

    private:
        State() = default;
        State(const State &) = delete;
        State &operator=(const State &) = delete;
        State(State &&) = delete;
        State &operator=(State &&) = delete;

        mutable std::shared_mutex mutex_;
        std::unordered_map<std::string, StateValue> state_;
        std::unordered_map<std::string, std::vector<StateCallback>> observers_;

        void notifyObservers(const std::string_view key, const std::optional<StateValue> &value) const {
            std::shared_lock lock(mutex_);
            if (const auto it = observers_.find(std::string(key)); it != observers_.end()) {
                for (const auto &callback: it->second) {
                    if (value) {
                        callback(key, *value);
                    } else {
                        callback(key, StateValue{});
                    }
                }
            }
        }

        template<typename T>
        static constexpr bool is_state_value_type_v =
                std::is_same_v<std::decay_t<T>, int> || std::is_same_v<std::decay_t<T>, double> ||
                std::is_same_v<std::decay_t<T>, std::string> || std::is_same_v<std::decay_t<T>, bool>;
    };

    // Convenience functions

    template<typename T>
    void set(std::string_view key, T &&value) {
        State::instance().set(key, std::forward<T>(value));
    }

    template<typename T>
    [[nodiscard]] auto get(const std::string_view key, T &&default_value) {
        if constexpr (std::is_convertible_v<T, std::string_view>) {
            return State::instance().get<std::string>(key, std::string(default_value));
        } else {
            return State::instance().get<std::decay_t<T>>(key, std::forward<T>(default_value));
        }
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
