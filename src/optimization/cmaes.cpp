#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <algorithm>
#include <functional>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class CMAES {
public:
    CMAES(int dimensions, int populationSize, double lowerBound, double upperBound) :
        dimensions_(dimensions),
        populationSize_(populationSize),
        lowerBound_(lowerBound),
        upperBound_(upperBound),
        mean_(VectorXd::Zero(dimensions)),
        covarianceMatrix_(MatrixXd::Identity(dimensions, dimensions)),
        stepSize_(0.5),
        randomEngine_(std::random_device{}()),
        normalDistribution_(0.0, 1.0),
        maxStepSize_(1.0),
        minStepSize_(1e-10),
        stagnationThreshold_(20),
        stagnationCount_(0),
        bestValue_(std::numeric_limits<double>::infinity()) {
        initialize();
    }

    void optimize(int generations, std::function<double(const VectorXd &)> objectiveFunction) {
        for (int generation = 0; generation < generations; ++generation) {
            populate();
            evaluate(objectiveFunction);
            evolve();
            prune();
            learn();

            if (generation % 10 == 0) {
                std::cout << "Generation " << generation << " Best Objective Value: " << bestValue_ << std::endl;
            }
        }
    }

    VectorXd getBestSolution() const {
        return bestSolution_;
    }

private:
    int dimensions_;
    int populationSize_;
    double lowerBound_;
    double upperBound_;
    VectorXd mean_;
    MatrixXd covarianceMatrix_;
    double stepSize_;
    double maxStepSize_;
    double minStepSize_;
    int stagnationThreshold_;
    int stagnationCount_;
    std::mt19937 randomEngine_;
    std::normal_distribution<double> normalDistribution_;

    std::vector<VectorXd> population_;
    std::vector<double> fitness_;
    VectorXd bestSolution_;
    double bestValue_;

    void initialize() {
        population_.resize(populationSize_);
        fitness_.resize(populationSize_);
    }

    void populate() {
        for (int i = 0; i < populationSize_; ++i) {
            population_[i] = mean_ + stepSize_ * sampleMultivariateNormal();
            for (int j = 0; j < dimensions_; ++j) {
                population_[i](j) = std::clamp(population_[i](j), lowerBound_, upperBound_);
            }
        }
    }

    VectorXd sampleMultivariateNormal() {
        VectorXd z(dimensions_);
        for (int i = 0; i < dimensions_; ++i) {
            z(i) = normalDistribution_(randomEngine_);
        }
        return covarianceMatrix_.llt().matrixL() * z;
    }

    void evaluate(std::function<double(const VectorXd &)> objectiveFunction) {
        for (int i = 0; i < populationSize_; ++i) {
            fitness_[i] = objectiveFunction(population_[i]);
            if (fitness_[i] < bestValue_) {
                bestValue_ = fitness_[i];
                bestSolution_ = population_[i];
                stagnationCount_ = 0;
            }
        }
    }

    void evolve() {
        VectorXd weightedSum = VectorXd::Zero(dimensions_);
        double fitnessSum = 0;
        for (int i = 0; i < populationSize_; ++i) {
            weightedSum += fitness_[i] * population_[i];
            fitnessSum += fitness_[i];
        }
        mean_ = weightedSum / fitnessSum;
    }

    void prune() {
        MatrixXd covUpdate = MatrixXd::Zero(dimensions_, dimensions_);
        for (int i = 0; i < populationSize_; ++i) {
            VectorXd deviation = population_[i] - mean_;
            covUpdate += deviation * deviation.transpose();
        }
        covarianceMatrix_ = covUpdate / populationSize_;
    }

    void learn() {
        stagnationCount_++;
        if (stagnationCount_ > stagnationThreshold_) {
            restart();
            stagnationCount_ = 0;
        } else {
            double learningRate = 1.0 / dimensions_;
            stepSize_ = std::max(minStepSize_, stepSize_ * std::exp(learningRate * (fitness_[0] - fitness_.back())));
        }
    }

    void restart() {
        mean_ = bestSolution_;
        covarianceMatrix_ = MatrixXd::Identity(dimensions_, dimensions_);
        stepSize_ = maxStepSize_ / 2;
        addDiversity();
    }

    void addDiversity() {
        for (int i = 0; i < dimensions_; ++i) {
            mean_(i) += normalDistribution_(randomEngine_) * stepSize_;
        }
    }
};

