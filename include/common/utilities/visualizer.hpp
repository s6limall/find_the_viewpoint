// File: common/utilities/visualizer.hpp

#ifndef VISUAL_HPP
#define VISUAL_HPP

#include <vector>
#include <random>

#include <pcl/visualization/pcl_visualizer.h>

#include "types/viewpoint.hpp"

namespace common::utilities {

    class Visualizer {
    public:
        static void visualizeResults(const std::vector<ViewPoint<double> > &samples, double inner_radius,
                                     double outer_radius);

        static void visualizeClusters(const std::vector<ViewPoint<double> > &samples);

        static void visualizeViewpoints(const std::vector<Eigen::Matrix4d> &poses, double inner_radius,
                                        double outer_radius);
    };
}

#endif //VISUAL_HPP
