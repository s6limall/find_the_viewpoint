cmake_minimum_required(VERSION 3.22...3.30)
project(viewpoint_core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable C language if required by MPI
find_package(MPI QUIET)
if (MPI_FOUND)
    enable_language(C)
endif ()

# Find packages
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization)
#find_package(jsoncpp CONFIG REQUIRED)
find_package(Freetype REQUIRED)
find_package(spdlog REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)


# Collect include directories and libraries
set(PROJECT_INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        ${fmt_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${JSONCPP_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${SPDLOG_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)
set(PROJECT_LIBRARIES
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        jsoncpp_lib
        ${FREETYPE_LIBRARIES}
        ${Python3_LIBRARIES}
        spdlog::spdlog
        yaml-cpp
        fmt::fmt
)

# Collect source files and headers
file(GLOB_RECURSE PROJECT_HEADERS CONFIGURE_DEPENDS include/*.hpp)
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS src/*.cpp)

# Main library target
add_library(${PROJECT_NAME} STATIC ${PROJECT_HEADERS} ${PROJECT_SOURCES})

# Set include directories for the library
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${PROJECT_INCLUDE_DIRS}
)

# Link libraries to the library target
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_LIBRARIES})

# Executable target
add_executable(${PROJECT_NAME}_exe ${PROJECT_HEADERS} ${PROJECT_SOURCES})

# Set include directories for the executable
target_include_directories(${PROJECT_NAME}_exe
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${PROJECT_INCLUDE_DIRS}
)

# Link libraries to the executable target
target_link_libraries(${PROJECT_NAME}_exe PUBLIC ${PROJECT_LIBRARIES})

# Set the output name for the executable
set_target_properties(${PROJECT_NAME}_exe PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

# Enable parallel builds
if (POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif ()

message(STATUS "Configuration complete for ${PROJECT_NAME} version ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
