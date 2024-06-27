// File: filtering/filter.hpp

#ifndef FILTER_HPP
#define FILTER_HPP

#include <vector>
#include <memory>
#include <functional>
#include <spdlog/spdlog.h>

namespace filtering {

    class Filter {
    public:
        virtual ~Filter() = default;

        // Filter points based on a threshold
        [[nodiscard]] virtual std::vector<std::vector<double> > filter(const std::vector<std::vector<double> > &points,
                                                                       double threshold) const = 0;

        // Add a filter to the chain
        template<typename T, typename... Args>
        void addFilter(Args &&... args) {
            static_assert(std::is_base_of_v<Filter, T>, "T must be derived from Filter");
            auto filter = std::make_shared<T>(std::forward<Args>(args)...);
            spdlog::info("Adding filter: {}", filter->getName());
            filters_.emplace_back(filter);
        }


        [[nodiscard]] virtual std::string getName() const = 0;

    protected:
        std::vector<std::shared_ptr<Filter> > filters_;

        [[nodiscard]] std::vector<std::vector<double> > applyFilters(
                const std::vector<std::vector<double> > &points, double threshold) const;
    };

}

#endif // FILTER_HPP
