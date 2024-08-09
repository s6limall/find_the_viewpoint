// File: types/concepts.hpp

#ifndef CONCEPTS_HPP
#define CONCEPTS_HPP

#include <concepts>
#include <functional>

/*
 * Numeric Concepts
 */

template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template<typename T>
concept Integral = std::is_integral_v<T>;

template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<typename T>
concept Signed = std::is_signed_v<T>;

template<typename T>
concept Unsigned = std::is_unsigned_v<T>;


/*
 * Callable, Function Concepts
 */

template<typename F, typename... Args>
concept Callable = requires(F f, Args &&...args) { std::invoke(f, std::forward<Args>(args)...); };

template<typename F, typename R, typename... Args>
concept CallableReturning = requires(F f, Args &&...args) {
    { std::invoke(f, std::forward<Args>(args)...) } -> std::same_as<R>;
};

template<typename F>
concept FunctionPointer = std::is_function_v<std::remove_pointer_t<F>>;

/*
 * Utility Concepts
 */

template<typename T, typename U>
concept Same = std::same_as<T, U>;

template<typename Base, typename Derived>
concept DerivedFrom = std::derived_from<Derived, Base>;

#endif // CONCEPTS_HPP
