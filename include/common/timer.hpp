// File: common/timer.hpp

#ifndef TIMER_HPP
#define TIMER_HPP

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <utility>

class Timer {
public:
    explicit Timer(std::string name) :
        name_(std::move(name)), start_time_(std::chrono::high_resolution_clock::now()), stopped_(false) {}

    ~Timer() { stop(); }

    void stop() {
        if (!stopped_.exchange(true)) {
            const auto end_time = std::chrono::high_resolution_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_).count();
            LOG_INFO("Execution time of {}: {} microseconds", name_, duration);
        }
    }

    Timer(Timer &&other) noexcept :
        name_(std::move(other.name_)), start_time_(other.start_time_), stopped_(other.stopped_.exchange(true)) {}

    Timer &operator=(Timer &&other) noexcept {
        if (this != &other) {
            if (!stopped_.exchange(true)) {
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
};

#endif // TIMER_HPP
