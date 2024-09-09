// File: common/traits/optimization_traits.hpp

#ifndef OPTIMIZATION_TRAITS_HPP
#define OPTIMIZATION_TRAITS_HPP

#include <Eigen/Dense>
#include <concepts>
#include <types/concepts.hpp>
#include "optimization/gaussian/kernel/matern_52.hpp"

namespace optimization {

    template<typename K, typename T>
    concept IsKernel = std::is_base_of_v<kernel::Kernel<T>, K>;

    template<FloatingPoint T = double>
    using DefaultKernel = kernel::Matern52<T>;


    /*template<typename T, typename K>
    concept KernelConcept =
            requires(K k, const Eigen::Matrix<T, Eigen::Dynamic, 1> &x, const Eigen::Matrix<T, Eigen::Dynamic, 1> &y,
                     const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X, int param_index) {
                typename K::VectorType;
                typename K::MatrixType;

                { k.compute(x, y) } -> std::convertible_to<T>;
                { k.computeGramMatrix(X, X) } -> std::convertible_to<typename K::MatrixType>;
                { k.computeGramMatrix(X) } -> std::convertible_to<typename K::MatrixType>;
                { k.computeGradient(x, y) } -> std::convertible_to<typename K::VectorType>;
                { k.computeGradientMatrix(X, param_index) } -> std::convertible_to<typename K::MatrixType>;
                { k.setParameters(std::declval<typename K::VectorType>()) } -> std::same_as<void>;
                { k.getParameters() } -> std::convertible_to<typename K::VectorType>;
                { k.getParameterNames() } -> std::convertible_to<std::vector<std::string>>;
                { k.isStationary() } -> std::convertible_to<bool>;
                { k.isIsotropic() } -> std::convertible_to<bool>;
                { k.getKernelType() } -> std::convertible_to<std::string>;
                { k.getParameterCount() } -> std::convertible_to<int>;
            };

    template<FloatingPoint T = double>
    using DefaultKernel = kernel::Matern52<T>;*/

} // namespace optimization

#endif // OPTIMIZATION_TRAITS_HPP
