// File: viewpoint/generator.hpp

#ifndef GENERATOR_HPP
#define GENERATOR_HPP

#include "core/perception.hpp";
#include "sampling/sampler/fibonacci.hpp"
#include "viewpoint/provider.hpp"

namespace viewpoint {
    template<typename T = double>
    class Generator final : public Provider<T> {
    public:
        explicit Generator(T radius, const std::shared_ptr<FeatureExtractor> &extractor);

        std::vector<Image<T>> provision(size_t num_points) override;
        Image<T> next() override;

    private:
        T radius_;
        std::shared_ptr<FeatureExtractor> extractor_;
    };

    // Definitions

    template<typename T>
    Generator<T>::Generator(T radius, const std::shared_ptr<FeatureExtractor> &extractor) :
        radius_(radius), extractor_(extractor) {}

    template<typename T>
    std::vector<Image<T>> Generator<T>::provision(size_t num_points) {
        FibonacciLatticeSampler<T> sampler({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
        auto samples = sampler.generate(num_points);

        std::vector<Image<T>> images;
        images.reserve(samples.cols());
        for (int i = 0; i < samples.cols(); ++i) {
            const Eigen::Matrix<T, 3, 1> position = samples.col(i).template head<3>() * radius_;
            core::View view = core::View::fromPosition(position);
            ViewPoint<> viewpoint = ViewPoint<T>::fromView(view);
            Eigen::Matrix4d extrinsics = view.getPose();
            cv::Mat rendered_view = core::Perception::render(extrinsics);
            Image<T> image(rendered_view, extractor_);
            image.setViewPoint(ViewPoint<T>(position, 0.0, 1.0)); // Initial score and uncertainty
            images.push_back(image);
        }
        return images;
    }

    template<typename T>
    Image<T> Generator<T>::next() {
        // Implement logic to return the next image if needed
        return Image<T>();
    }
} // namespace viewpoint

#endif // GENERATOR_HPP
