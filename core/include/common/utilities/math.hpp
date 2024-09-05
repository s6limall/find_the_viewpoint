// File: common/utilities/math.hpp

#ifndef MATH_HPP
#define MATH_HPP

#include <type_traits>
#include <cmath>
#include <array>
#include <algorithm>
#include <numeric>
#include <ranges>

namespace common::utilities {
    template<typename T>
    constexpr bool isPrime(T number) noexcept {
        static_assert(std::is_integral_v<T>, "Type must be an integral type");

        if (number <= 1)
            return false;
        if (number <= 3)
            return true;
        if (number % 2 == 0 || number % 3 == 0)
            return number == 2 || number == 3;

        T limit = static_cast<T>(std::sqrt(number));

        for (T i = 5; i <= limit; i += 6) {
            if (number % i == 0 || number % (i + 2) == 0) {
                return false;
            }
        }
        return true;
    }

    template<typename T>
    constexpr T gcd(T a, T b) noexcept {
        static_assert(std::is_integral_v<T>, "Type must be an integral type");
        return std::gcd(a, b);
    }

    template<typename T>
    constexpr T lcm(T a, T b) noexcept {
        static_assert(std::is_integral_v<T>, "Type must be an integral type");
        return std::lcm(a, b);
    }

    template<typename T>
    constexpr T power(T base, T exponent) noexcept {
        static_assert(std::is_arithmetic_v<T>, "Type must be a numeric type");

        T result = 1;
        while (exponent > 0) {
            if (exponent % 2 == 1) {
                result *= base;
            }
            base *= base;
            exponent /= 2;
        }
        return result;
    }

    template<typename T>
    constexpr T factorial(T number) noexcept {
        static_assert(std::is_integral_v<T>, "Type must be an integral type");

        T result = 1;
        for (T i = 2; i <= number; ++i) {
            result *= i;
        }
        return result;
    }
}

#endif //MATH_HPP
