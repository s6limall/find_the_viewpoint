// File: common/formatting/fmt_fallback.hpp

#ifndef FMT_FALLBACK_HPP
#define FMT_FALLBACK_HPP

#include <fmt/core.h>
#include <sstream>
#include <string>

template<typename T>
struct fmt::formatter<T, char, std::enable_if_t<!fmt::has_formatter<T, fmt::format_context>::value> > {
    template<typename FormatContext>
    auto format(const T &value, FormatContext &ctx) {
        std::ostringstream oss;
        oss << value;
        return fmt::format_to(ctx.out(), "{}", oss.str());
    }
};

#endif // FMT_FALLBACK_HPP

