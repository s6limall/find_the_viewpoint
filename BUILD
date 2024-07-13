load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

# Define the project include directories
cc_library(
    name = "find_the_viewpoint_lib",
    srcs = glob(["src/**/*.cpp"]),
    hdrs = glob(["include/**/*.hpp`"]),
    copts = [
        "-O3",  # Optimize for speed
        "-march=native",  # Use the architecture of the local machine
        "-flto",  # Enable Link Time Optimization
        "-fno-omit-frame-pointer",  # Keep frame pointer for better debugging
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",  # Enable all warnings
    ],
    includes = ["include"],
    deps = [
        "@eigen",
        "@fmt",
        "@opencv",
        "@pcl",
        "@spdlog",
        "@yaml_cpp",
    ],
)

# Define the executable target
cc_binary(
    name = "find_the_viewpoint",
    srcs = ["src/main.cpp"],
    copts = [
        "-O3",
        "-march=native",
        "-flto",
        "-fno-omit-frame-pointer",
    ],
    deps = [
        ":find_the_viewpoint_lib",
    ],
)

# Add tests
cc_test(
    name = "tests",
    srcs = glob(["tests/**/*.cpp"]),
    copts = [
         "-O3",
        "-march=native",
        "-flto",
        "-fno-omit-frame-pointer",
    ],
    deps = [
        ":find_the_viewpoint_lib",
    ],
)
