// File: common/timer.hpp

#ifndef TIMER_HPP
#define TIMER_HPP

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <utility>

#include "logging/logger.hpp"

/* Destroys itself without requiring a manual call to stop() - via the destructor. */
class Timer {
public:
    explicit Timer(std::string name) :
        name_(std::move(name)), start_time_(std::chrono::high_resolution_clock::now()), stopped_(false) {
        LOG_DEBUG("Created timer for [{}]. Starting at {} ({} µs)", name_,
                  toHumanReadable(start_time_.time_since_epoch().count()),
                  std::chrono::duration_cast<std::chrono::microseconds>(start_time_.time_since_epoch()).count());
    }

    ~Timer() {
        if (!stopped_) { // Prevent unnecessary atomic operation if already stopped
            stop();
        }
    }


    // Stop the timer and log the duration
    void stop() noexcept {
        if (!stopped_.exchange(true)) {
            const auto end_time = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_).count();
            LOG_INFO("Execution time of {}: {} ({} µs).", name_, toHumanReadable(duration), duration);
        }
    }


    // Move constructor
    Timer(Timer &&other) noexcept :
        name_(std::move(other.name_)), start_time_(other.start_time_), stopped_(other.stopped_.exchange(true)) {}

    // Move assignment operator
    Timer &operator=(Timer &&other) noexcept {
        if (this != &other) {
            if (!stopped_) { // Stop current timer if not already stopped
                stop();
            }
            name_ = std::move(other.name_);
            start_time_ = other.start_time_;
            stopped_ = other.stopped_.exchange(true);
        }
        return *this;
    }

    // Deleted copy constructor and assignment operator to prevent copying
    Timer(const Timer &) = delete;
    Timer &operator=(const Timer &) = delete;


private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::atomic<bool> stopped_;

    static std::string toHumanReadable(const int64_t micros) {
        using namespace std::chrono;

        auto duration = microseconds(micros);
        const auto h = duration_cast<hours>(duration);
        duration -= h;
        const auto m = duration_cast<minutes>(duration);
        duration -= m;
        const auto s = duration_cast<seconds>(duration);
        duration -= s;
        const auto ms = duration_cast<milliseconds>(duration);
        duration -= ms;

        // Utilize std::format with conditional inclusion of time units
        return std::format("{}{}{}{}{}µs", h.count() > 0 ? std::format("{}h ", h.count()) : "",
                           m.count() > 0 || h.count() > 0 ? std::format("{}m ", m.count()) : "",
                           s.count() > 0 || m.count() > 0 || h.count() > 0 ? std::format("{}s ", s.count()) : "",
                           ms.count() > 0 || s.count() > 0 || m.count() > 0 || h.count() > 0
                                   ? std::format("{}ms ", ms.count())
                                   : "",
                           duration.count());
    }
};

#endif // TIMER_HPP
