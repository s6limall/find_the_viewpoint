// File: executor.hpp

#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include <memory>
#include <mutex>

#include "common/state/state.hpp"
#include "common/traits/optimization_traits.hpp"
#include "common/utilities/visualizer.hpp"
#include "misc/target_generator.hpp"
#include "optimization/gaussian/gpr.hpp"
#include "optimization/viewpoint_optimizer.hpp"
#include "processing/image/comparator.hpp"
#include "processing/image/feature/matcher/flann_matcher.hpp"
#include "processing/vision/estimation/distance_estimator.hpp"
#include "sampling/sampler/fibonacci.hpp"
#include "types/image.hpp"

class Executor {
public:
    using KernelType = optimization::DefaultKernel<double>;

    static void execute();
    static void reset();

    // Rule of five
    Executor(const Executor &) = delete;
    Executor &operator=(const Executor &) = delete;
    Executor(Executor &&) = delete;
    Executor &operator=(Executor &&) = delete;
    ~Executor() = default;

    // Delete the default constructor (no need to instantiate this class)
    Executor() = delete;

private:
    static std::once_flag init_flag_;
    static double radius_, target_score_;
    static Image<> target_;
    static std::shared_ptr<processing::image::ImageComparator> comparator_;
    static std::shared_ptr<processing::image::FeatureExtractor> extractor_;
    static std::shared_ptr<processing::image::FeatureMatcher> matcher_;
    static std::shared_ptr<KernelType> kernel_;
    static std::shared_ptr<optimization::GPR<>> gpr_;
    static std::shared_ptr<optimization::ViewpointOptimizer<>> optimizer_;

    static void initialize();
};

#endif // EXECUTOR_HPP
