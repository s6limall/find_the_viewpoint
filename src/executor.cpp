// File: executor.cpp

#include "executor.hpp"

using common::utilities::Visualizer;

std::once_flag Executor::init_flag_;
cv::Mat Executor::target_image_;

void Executor::initialize() {
    target_image_ = common::io::image::readImage(config::get("paths.target_image", "target.png"));

    // Create instances of feature extractor and matcher
    auto extractor = processing::image::FeatureExtractor::create<processing::image::ORBExtractor>();
    auto matcher = processing::image::FeatureMatcher::create<processing::image::BFMatcher>();

    const auto sift_extractor = processing::image::FeatureExtractor::create<processing::image::SIFTExtractor>();
    auto [keypoints, descriptors] = sift_extractor->extract(target_image_);

    const auto camera = std::make_shared<core::Camera>();
    const auto width = config::get("camera.width", Defaults::width);
    const auto height = config::get("camera.height", Defaults::height);
    const auto fov_x = config::get("camera.fov_x", Defaults::fov_x);
    const auto fov_y = config::get("camera.fov_y", Defaults::fov_y);

    LOG_INFO("Configuring camera with width={}, height={}, fov_x={}, fov_y={}", width, height, fov_x, fov_y);
    camera->setIntrinsics(width, height, fov_x, fov_y);

    LOG_INFO("Executor initialized.");
}

void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);

    const double distance = estimateDistance();
    auto samples = generateSamples(distance);

    const processing::image::ImageComparator &ssim = processing::image::SSIMComparator();
    auto evaluated_samples = evaluateSamples(ssim, samples, distance);

    QuadrantFilter<ViewPoint<double> > quadrant_filter;
    auto filtered_samples = quadrant_filter.filter(evaluated_samples, [](const ViewPoint<double> &view_point) {
        return view_point.getScore();
    });

    auto clusters = clusterSamples(filtered_samples);

    std::vector<ViewPoint<double> > points;
    for (const auto &cluster: clusters) {
        points.insert(points.end(), cluster.getPoints().begin(), cluster.getPoints().end());
        LOG_INFO("Points in cluster: {}, Average score = {}", cluster.size(), cluster.getAverageScore());
    }

    const Cluster<double> &best_cluster = *std::max_element(clusters.begin(), clusters.end(),
                                                            [](const Cluster<double> &a, const Cluster<double> &b) {
                                                                return a.getAverageScore() < b.getAverageScore();
                                                            });


    LOG_INFO("Best cluster: {}, Average score = {}", best_cluster.size(), best_cluster.getAverageScore());
    Visualizer::visualizeResults(best_cluster.getPoints(), distance * 0.9, distance * 1.1);
}


double Executor::estimateDistance() {
    const size_t max_iterations = config::get("estimation.distance.max_iterations", 20);
    const double initial_distance = config::get("estimation.distance.initial_guess", 1.0);
    return DistanceEstimator::estimateDistance(target_image_, max_iterations, initial_distance);
}

std::vector<ViewPoint<double> > Executor::generateSamples(const double estimated_distance) {
    const size_t num_samples = config::get("sampling.count", 100);

    sampling::SphericalShellTransformer transformer(estimated_distance * 0.9, estimated_distance * 1.1);
    sampling::HaltonSampler haltonSampler([&transformer](const std::vector<double> &sample) {
        return transformer.transform(sample);
    });
    auto halton_sequence = haltonSampler.generate(num_samples,
                                                  {0.0, 0.0, 0.0},
                                                  {1.0, 1.0, 1.0});

    std::vector<ViewPoint<double> > samples;
    samples.reserve(halton_sequence.size());
    for (const auto &data: halton_sequence) {
        samples.emplace_back(data[0], data[1], data[2]);
    }

    return samples;
}

std::vector<ViewPoint<double> > Executor::evaluateSamples(const processing::image::ImageComparator &comparator,
                                                          const std::vector<ViewPoint<double> > &samples,
                                                          const double distance) {
    std::vector<ViewPoint<double> > evaluated_samples;
    evaluated_samples.reserve(samples.size());

    for (const auto &sample: samples) {
        core::View view = sample.toView();
        cv::Mat rendered_image = core::Perception::render(view.getPose());
        const double score = comparator.compare(target_image_, rendered_image);
        auto evaluated_sample = sample;
        evaluated_sample.setScore(score);
        evaluated_samples.push_back(evaluated_sample);
    }

    return evaluated_samples;
}

std::vector<Cluster<double> > Executor::clusterSamples(const std::vector<ViewPoint<double> > &evaluated_samples) {
    auto metric = [](const ViewPoint<double> &a, const ViewPoint<double> &b) {
        return std::abs(a.getScore() - b.getScore());
    };

    clustering::DBSCAN<double> dbscan(5, metric); // min_points = 5
    std::vector<ViewPoint<double> > samples = evaluated_samples;
    auto clusters = dbscan.cluster(samples);

    // Logging cluster information
    for (const auto &cluster: clusters) {
        double average_score = cluster.getAverageScore();
        LOG_DEBUG("Cluster ID: {}, Number of Points: {}, Average Score: {}", cluster.getClusterId(), cluster.size(),
                  average_score);
        for (const auto &point: cluster.getPoints()) {
            auto [x, y, z] = point.toCartesian();
            LOG_DEBUG("Point: ({}, {}, {}), Score: {}", x, y, z, point.getScore());
        }
    }

    return clusters;
}


/*
std::vector<ViewPoint<double> > Executor::clusterSamples(const std::vector<ViewPoint<double> > &evaluated_samples) {
    auto ssimMetric = [](const ViewPoint<double> &a, const ViewPoint<double> &b) {
        return std::abs(a.getScore() - b.getScore());
    };
    clustering::DBSCAN<double> dbscan(5, ssimMetric); // min_points = 5
    std::vector<ViewPoint<double> > samples = evaluated_samples;
    dbscan.cluster(samples);

    // Logging cluster information
    std::map<int, std::vector<ViewPoint<double> > > clusters;
    for (const auto &sample: samples) {
        clusters[sample.getClusterId()].push_back(sample);
    }

    for (const auto &[id, points]: clusters) {
        double average_score = std::accumulate(points.begin(), points.end(), 0.0,
                                               [](const double sum, const ViewPoint<double> &vp) {
                                                   return sum + vp.getScore();
                                               }) / static_cast<double>(points.size());
        LOG_DEBUG("Cluster ID: {}, Number of Points: {}, Average Score: {}", id, points.size(), average_score);
        for (const auto &point: points) {
            LOG_DEBUG("Point: ({}, {}, {}), Score: {}", point.getPosition().x(), point.getPosition().y(),
                      point.getPosition().z(), point.getScore());
        }
    }

    return samples;
}
*/

/*
void Executor::execute() {
    std::call_once(init_flag_, &Executor::initialize);


    const double distance = estimateDistance();
    const auto samples = generateSamples(distance);

    const processing::image::ImageComparator &ssim = processing::image::SSIMComparator();

    const auto evaluated_samples = evaluateSamples(ssim, samples, distance);

    Visualizer::visualizeResults(samples, distance * 0.9, distance * 1.1);

    const auto clusters = clusterSamples(evaluated_samples);

    std::vector<ViewPoint<double> > points;
    for (const auto &cluster: clusters) {
        points.insert(points.end(), cluster.getPoints().begin(), cluster.getPoints().end());
        LOG_INFO("Points in cluster: {}, Average score = {}, ", cluster.size(), cluster.getAverageScore());
    }


    Visualizer::visualizeClusters(points);

    const processing::image::ImageComparator &feature_comparator = processing::image::FeatureComparator();
    const auto second_eval = evaluateSamples(feature_comparator, points, distance);
    Visualizer::visualizeResults(second_eval, distance * 0.9, distance * 1.1);
    const auto second_clusters = clusterSamples(second_eval);
    std::vector<ViewPoint<double> > second_points;
    for (const auto &cluster: second_clusters) {
        second_points.insert(second_points.end(), cluster.getPoints().begin(), cluster.getPoints().end());
        LOG_INFO("Points in cluster: {}, Average score = {}, ", cluster.size(), cluster.getAverageScore());
    }
    Visualizer::visualizeClusters(second_points);

}
 */
