// File: viewpoint/generator.hpp

#ifndef VIEWPOINT_GENERATOR_HPP
#define VIEWPOINT_GENERATOR_HPP

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "core/view.hpp"
#include "viewpoint/provider.hpp"
#include "config/configuration.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"
#include "sampling/sampler/halton_sampler.hpp"
#include "sampling/transformer/spherical_transformer.hpp"

namespace viewpoint {

    template<typename T = double>
    class Generator final : public Provider<T> {
    public:
        explicit Generator(const double distance) :
            distance_(distance) {
        }

        std::vector<ViewPoint<T> > provision() override;

    private:
        double distance_;
    };

    template<typename T>
    std::vector<ViewPoint<T> > Generator<T>::provision() {
        const size_t num_samples = config::get("sampling.count", 100);

        sampling::SphericalShellTransformer transformer(distance_ * 0.9, distance_ * 1.1);
        sampling::HaltonSampler haltonSampler([&transformer](const std::vector<double> &sample) {
            return transformer.transform(sample);
        });
        auto halton_sequence = haltonSampler.generate(num_samples,
                                                      {0.0, 0.0, 0.0},
                                                      {1.0, 1.0, 1.0});

        std::vector<ViewPoint<> > samples;
        samples.reserve(halton_sequence.size());
        for (const auto &data: halton_sequence) {
            samples.emplace_back(data[0], data[1], data[2]);
        }

        return samples;
    }

}

#endif // VIEWPOINT_GENERATOR_HPP
