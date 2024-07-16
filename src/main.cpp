// main.cpp

#include "executor.hpp"

// File: main.cpp

#include <iostream>
#include "optimization/cmaes.hpp"
#include "types/viewpoint.hpp"
#include "viewpoint/evaluator.hpp"
#include "processing/image/comparator.hpp"
#include "common/io/image.hpp"

double evaluateViewpoint(const ViewPoint<double> &viewpoint, viewpoint::Evaluator<> &evaluator,
                         const std::unique_ptr<processing::image::ImageComparator> &comparator) {
    // Wrap the viewpoint in a vector for compatibility with the evaluator
    std::vector<ViewPoint<double> > sample{viewpoint};
    auto evaluated_images = evaluator.evaluate(comparator, sample);

    // Return the score of the single evaluated image
    return evaluated_images[0].getScore();
}

int main() {
    // Load target image and create evaluator
    auto target_image = common::io::image::readImage("../../task1/target_images/obj_000020/img.png");
    auto extractor = FeatureExtractor::create<processing::image::SIFTExtractor>();
    Image<> target(target_image, extractor);
    auto comparator = std::make_unique<processing::image::SSIMComparator>();
    auto evaluator = std::make_unique<viewpoint::Evaluator<> >(target);

    // Define the evaluation function
    auto evaluation_function = [&](const ViewPoint<double> &viewpoint) {
        return evaluateViewpoint(viewpoint, *evaluator, comparator);
    };

    // Initial guess and bounds for optimization
    std::vector<double> initial_guess = {1.0, 1.0, 1.0};
    std::vector<double> lower_bounds = {-10.0, -10.0, -10.0};
    std::vector<double> upper_bounds = {10.0, 10.0, 10.0};

    // Create and run the optimizer
    ViewpointOptimizer optimizer(evaluation_function);
    ViewPoint<double> optimal_viewpoint = optimizer.optimize(initial_guess, lower_bounds, upper_bounds);

    std::cout << "Optimal Viewpoint: " << optimal_viewpoint.toString() << std::endl;

    return 0;
}

/*
int main() {

    try {
        Executor::execute();
    } catch (const std::exception &e) {
        LOG_ERROR("An error occurred during execution.", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
*/
