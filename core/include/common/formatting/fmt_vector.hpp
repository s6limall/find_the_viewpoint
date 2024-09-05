// File: common/formatting/fmt_vector.hpp

#ifndef COMMON_FORMATTING_FMT_VECTOR_HPP
#define COMMON_FORMATTING_FMT_VECTOR_HPP

#include <vector>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <type_traits>

// Helper trait to check if a type is a specialization of std::vector
template<typename T>
struct is_vector : std::false_type {
};

template<typename T, typename Alloc>
struct is_vector<std::vector<T, Alloc> > : std::true_type {
};

// Custom formatter for std::vector
template<typename T>
struct fmt::formatter<T, std::enable_if_t<is_vector<T>::value> > {
    // Parse function required by fmt
    constexpr auto parse(fmt::format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    // Format function required by fmt
    template<typename FormatContext>
    auto format(const T &vec, FormatContext &ctx) const -> decltype(ctx.out()) {
        auto out = ctx.out();
        *out++ = '[';

        for (size_t i = 0; i < vec.size(); ++i) {
            if (i != 0) {
                *out++ = ',';
                *out++ = ' ';
            }
            fmt::format_to(out, "{}", vec[i]);
        }

        *out++ = ']';
        return out;
    }
};

// Specialization for nested vectors
template<typename T>
struct fmt::formatter<std::vector<T> > {
    constexpr auto parse(fmt::format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }

    template<typename FormatContext>
    auto format(const std::vector<T> &vec, FormatContext &ctx) const -> decltype(ctx.out()) {
        auto out = ctx.out();
        *out++ = '[';

        for (size_t i = 0; i < vec.size(); ++i) {
            if (i != 0) {
                *out++ = ',';
                *out++ = ' ';
            }
            fmt::format_to(out, "{}", vec[i]);
        }

        *out++ = ']';
        return out;
    }
};

#endif // COMMON_FORMATTING_FMT_VECTOR_HPP

