// common/formatting/fmt_vector.hpp

#ifndef FMT_VECTOR_HPP
#define FMT_VECTOR_HPP

#include <fmt/core.h>
#include <vector>
#include <string>

// Generic formatter for std::vector<T>
template<typename T>
struct fmt::formatter<std::vector<T> > {
    constexpr auto parse(fmt::format_parse_context &ctx) -> decltype(ctx.begin()) {
        // We can add parsing logic here if needed
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const std::vector<T> &vec, FormatContext &ctx) -> decltype(ctx.out()) {
        auto out = ctx.out();
        fmt::format_to(out, "[");
        for (size_t i = 0; i < vec.size(); ++i) {
            if constexpr (std::is_same_v<T, std::string>) {
                fmt::format_to(out, "\"{}\"", vec[i]); // Special case for std::string
            } else {
                fmt::format_to(out, "{}", vec[i]); // Generic case for other types
            }
            if (i < vec.size() - 1) {
                fmt::format_to(out, ", ");
            }
        }
        return fmt::format_to(out, "]");
    }
};

#endif // FMT_VECTOR_HPP
