// File: optimization/kernel/kernel.hpp

#ifndef KERNEL_HPP
#define KERNEL_HPP

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>
#include "types/concepts.hpp"

namespace optimization::kernel {

    template<FloatingPoint T = double>
    class Kernel {
    public:
        using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
        using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        virtual ~Kernel() = default;

        // Core functionality
        [[nodiscard]] virtual T compute(const VectorType &x, const VectorType &y) const = 0;
        [[nodiscard]] virtual MatrixType computeGramMatrix(const MatrixType &X, const MatrixType &Y) const = 0;
        [[nodiscard]] virtual MatrixType computeGramMatrix(const MatrixType &X) const = 0;

        // Gradient computation
        [[nodiscard]] virtual VectorType computeGradient(const VectorType &x, const VectorType &y) const = 0;
        [[nodiscard]] virtual MatrixType computeGradientMatrix(const MatrixType &X, int param_index) const = 0;

        // Parameters
        virtual void setParameters(const VectorType &params) = 0;
        [[nodiscard]] virtual VectorType getParameters() const = 0;
        [[nodiscard]] virtual std::vector<std::string> getParameterNames() const = 0;

        // Kernel properties
        [[nodiscard]] virtual bool isStationary() const = 0;
        [[nodiscard]] virtual bool isIsotropic() const = 0;

        [[nodiscard]] virtual std::shared_ptr<Kernel<T>> clone() const = 0;
        [[nodiscard]] virtual int getParameterCount() const = 0;

        // TODO: unused, remove?
        [[nodiscard]] virtual MatrixType computeBasisFunctions(const MatrixType &X) const {
            throw std::runtime_error("Basis functions not implemented for this kernel");
        }

        [[nodiscard]] virtual VectorType computeEigenvalues(int num_eigenvalues) const {
            throw std::runtime_error("Eigenvalue computation not implemented for this kernel");
        }


        // Utility functions
        [[nodiscard]] virtual std::string getKernelType() const = 0;

    protected:
        static void validateParameters(const VectorType &params, const std::vector<std::string> &param_names) {
            if (params.size() != static_cast<size_t>(param_names.size())) {
                throw std::invalid_argument("Number of parameters does not match expected number for this kernel");
            }
        }
    };

} // namespace optimization::kernel

#endif // KERNEL_HPP
