# cmake/Dependencies.cmake

# Find necessary packages
#find_package(OpenCV REQUIRED)
#find_package(PCL REQUIRED COMPONENTS common io visualization)
#find_package(Eigen3 REQUIRED NO_MODULE)
#find_package(jsoncpp CONFIG REQUIRED)
#find_package(Freetype REQUIRED)
#find_package(fmt REQUIRED)
#find_package(spdlog REQUIRED)
#find_package(yaml-cpp REQUIRED)

# Check if the dependencies have already been included to avoid conflicts
if (NOT TARGET OpenCV)
    find_package(OpenCV REQUIRED)
endif ()

if (NOT TARGET PCL)
    find_package(PCL REQUIRED COMPONENTS common io visualization)
endif ()

if (NOT TARGET Eigen3)
    find_package(Eigen3 REQUIRED NO_MODULE)
endif ()

if (NOT TARGET JsonCpp)
    find_package(jsoncpp CONFIG REQUIRED)
endif ()

if (NOT TARGET Freetype)
    find_package(Freetype REQUIRED)
endif ()

if (NOT TARGET fmt)
    find_package(fmt REQUIRED)
endif ()

if (NOT TARGET spdlog)
    find_package(spdlog REQUIRED)
endif ()

if (NOT TARGET yaml-cpp)
    find_package(yaml-cpp REQUIRED)
endif ()

# Set include directories and libraries for the project
set(INCLUDE_DIRS
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIRS}
        ${JSONCPP_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${SPDLOG_INCLUDE_DIRS}
)

set(LIBRARIES
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        ${FREETYPE_LIBRARIES}
        spdlog::spdlog
        yaml-cpp
        fmt::fmt
)
