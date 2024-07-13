// tests/sampling/halton_sampler_test.cpp

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "sampling/halton_sampler.hpp"

using namespace sampling;
using ::testing::ElementsAreArray;

class HaltonSamplerTest : public ::testing::Test {
protected:
    HaltonSampler sampler;

    std::vector<double> lower_bounds = {0.0, 0.0, 0.0};
    std::vector<double> upper_bounds = {1.0, 1.0, 1.0};


};

TEST_F(HaltonSamplerTest, GenerateSamples) {
    auto samples = sampler.generate(10, lower_bounds, upper_bounds);
    EXPECT_EQ(samples.size(), 10);
    for (const auto& sample : samples) {
        EXPECT_EQ(sample.size(), lower_bounds.size());
        for (size_t i = 0; i < sample.size(); ++i) {
            EXPECT_GE(sample[i], lower_bounds[i]);
            EXPECT_LE(sample[i], upper_bounds[i]);
        }
    }
}

TEST_F(HaltonSamplerTest, NextSample) {
    sampler.generate(10, lower_bounds, upper_bounds);
    auto sample = sampler.next();
    EXPECT_EQ(sample.size(), lower_bounds.size());
    for (size_t i = 0; i < sample.size(); ++i) {
        EXPECT_GE(sample[i], lower_bounds[i]);
        EXPECT_LE(sample[i], upper_bounds[i]);
    }
}

TEST_F(HaltonSamplerTest, ExceptionOnGenerateWithZeroSamples) {
    EXPECT_THROW(sampler.generate(0, lower_bounds, upper_bounds), std::invalid_argument);
}

TEST_F(HaltonSamplerTest, ExceptionOnNextWithoutGenerate) {
    EXPECT_THROW(sampler.next(), std::runtime_error);
}

TEST_F(HaltonSamplerTest, AdaptiveMode) {
    auto adapt_function = [](std::vector<double>& sample) {
        for (auto& value : sample) {
            value *= 0.5;
        }
    };

    sampler.setAdaptive(true, adapt_function);
    auto samples = sampler.generate(5, lower_bounds, upper_bounds);

    for (const auto& sample : samples) {
        for (const auto& value : sample) {
            EXPECT_LE(value, 0.5);
        }
    }
}

TEST_F(HaltonSamplerTest, CalculateDiscrepancy) {
    auto samples = sampler.generate(10, lower_bounds, upper_bounds);
    double discrepancy = Sampler::calculateDiscrepancy(samples);
    EXPECT_GT(discrepancy, 0.0);
    EXPECT_LE(discrepancy, 1.0);
}

TEST_F(HaltonSamplerTest, Discrepancy) {
    sampler.generate(10, lower_bounds, upper_bounds);
    double discrepancy = sampler.discrepancy();
    EXPECT_GT(discrepancy, 0.0);
    EXPECT_LE(discrepancy, 1.0);
}

TEST_F(HaltonSamplerTest, ValidateBoundsException) {
    std::vector<double> invalid_lower_bounds = {1.0, 1.0, 1.0};
    std::vector<double> invalid_upper_bounds = {0.0, 0.0, 0.0};
    EXPECT_THROW(sampler.generate(10, invalid_lower_bounds, invalid_upper_bounds), std::invalid_argument);
}

TEST_F(HaltonSamplerTest, DefaultAdaptFunction) {
    sampler.setAdaptive(true);
    auto samples = sampler.generate(5, lower_bounds, upper_bounds);

    for (const auto& sample : samples) {
        for (const auto& value : sample) {
            EXPECT_GE(value, lower_bounds[0]);
            EXPECT_LE(value, upper_bounds[0]);
        }
    }
}

TEST_F(HaltonSamplerTest, ResetSampler) {
    sampler.generate(10, lower_bounds, upper_bounds);
    sampler.reset();
    auto samples = sampler.generate(5, lower_bounds, upper_bounds);
    EXPECT_EQ(samples.size(), 5);
}

