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
            distance_(distance),
            halton_sampler_([this](const std::vector<double> &sample) {
                return transformer_.transform(sample);
            }),
            transformer_(distance_ * 0.9, distance_ * 1.1) {
        }

        std::vector<ViewPoint<T> > provision() override;

        ViewPoint<T> next() override;

    private:
        double distance_;
        sampling::HaltonSampler halton_sampler_;
        sampling::SphericalShellTransformer transformer_;
    };

    template<typename T>
    std::vector<ViewPoint<T> > Generator<T>::provision() {
        const size_t num_samples = config::get("sampling.count", 100);

        auto halton_sequence = halton_sampler_.generate(num_samples,
                                                        {0.0, 0.0, 0.0},
                                                        {1.0, 1.0, 1.0});

        std::vector<ViewPoint<T> > samples;
        samples.reserve(halton_sequence.size());
        for (const auto &data: halton_sequence) {
            samples.emplace_back(data[0], data[1], data[2]);
        }

        return samples;
    }

    template<typename T>
    ViewPoint<T> Generator<T>::next() {
        auto sample = halton_sampler_.next();
        return ViewPoint<T>(sample[0], sample[1], sample[2]);
    }

}

#endif // VIEWPOINT_GENERATOR_HPP
