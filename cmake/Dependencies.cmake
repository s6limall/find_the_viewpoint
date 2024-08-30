# File: cmake/Dependencies.cmake

# Find and configure required packages
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui xfeatures2d quality dnn xphoto ximgproc photo)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(fmt REQUIRED)

# Handle JsonCpp separately to avoid conflicts
if(NOT TARGET JsonCpp::JsonCpp)
    find_package(jsoncpp CONFIG QUIET)
    if(NOT jsoncpp_FOUND)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(JSONCPP jsoncpp)
        if(JSONCPP_FOUND)
            add_library(JsonCpp::JsonCpp INTERFACE IMPORTED)
            set_target_properties(JsonCpp::JsonCpp PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${JSONCPP_INCLUDE_DIRS}"
                    INTERFACE_LINK_LIBRARIES "${JSONCPP_LIBRARIES}"
            )
        else()
            message(FATAL_ERROR "JsonCpp not found")
        endif()
    endif()
endif()

# Aggregate include directories
set(PROJECT_INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${spdlog_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        ${fmt_INCLUDE_DIRS}
)

# Add JsonCpp include directories if found
if(JSONCPP_INCLUDE_DIRS)
    list(APPEND PROJECT_INCLUDE_DIRS ${JSONCPP_INCLUDE_DIRS})
endif()

# Aggregate libraries
set(PROJECT_LIBRARIES
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        Eigen3::Eigen
        spdlog::spdlog
        yaml-cpp
        fmt::fmt
)

# Add JsonCpp to libraries
if(TARGET JsonCpp::JsonCpp)
    list(APPEND PROJECT_LIBRARIES JsonCpp::JsonCpp)
elseif(JSONCPP_LIBRARIES)
    list(APPEND PROJECT_LIBRARIES ${JSONCPP_LIBRARIES})
endif()