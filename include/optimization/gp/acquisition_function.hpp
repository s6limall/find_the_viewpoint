#ifndef ACQUISITION_FUNCTION_HPP
#define ACQUISITION_FUNCTION_HPP

#include <cmath>
#include <algorithm>

template<typename GP, typename T>
class AcquisitionFunction {
public:
    double operator()(const GP &gp, const T &point) const {
        double mu = gp.mean(point);
        double sigma = std::sqrt(gp.covariance(point, point));
        double best_value = *std::max_element(gp.getValues().begin(), gp.getValues().end());

        const double kappa = 2.0;
        return mu + kappa * sigma;
    }
};

#endif // ACQUISITION_FUNCTION_HPP
