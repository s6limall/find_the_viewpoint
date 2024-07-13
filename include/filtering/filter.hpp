// File: core/filter.hpp

#ifndef FILTER_HPP
#define FILTER_HPP

#include <vector>

template<typename T>
class Filter {
public:
    virtual ~Filter() = default;

    virtual std::vector<T> filter(const std::vector<T> &samples,
                                  std::function<double(const T &)> evaluation_function) = 0;
};

#endif // FILTER_HPP
