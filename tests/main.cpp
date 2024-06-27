// tests/main.cpp

#include <gtest/gtest.h>

// Test runner
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "Hello";
    return RUN_ALL_TESTS();
}
