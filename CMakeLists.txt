cmake_minimum_required(VERSION 3.14...3.30)
project(ftv VERSION 1.0.0 LANGUAGES CXX)

include(cmake/ProjectSettings.cmake)
include(cmake/CompilerSettings.cmake)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(xarm_msgs REQUIRED)
find_package(xarm_api REQUIRED)
find_package(xarm_description REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo_ros_pkgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(xacro REQUIRED)
find_package(urdf REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)

# Find VTK without MPI
set(VTK_USE_MPI OFF CACHE BOOL "Use MPI" FORCE)
find_package(VTK REQUIRED COMPONENTS
        CommonCore
        CommonDataModel
        IOImage
        InteractionStyle
        RenderingCore
        RenderingOpenGL2
)

# Find PCL
find_package(PCL REQUIRED COMPONENTS common io visualization)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Include core library
add_subdirectory(core)

# Use globbing to find all source files in src/
file(GLOB_RECURSE SRC_FILES src/*.cpp)

# Use globbing to find all header files in include/
file(GLOB_RECURSE INCLUDE_FILES include/*.hpp include/*.h)

# Add executable
add_executable(ftv_ros src/main.cpp)

# Set include directories for the executable
target_include_directories(ftv_ros PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CMAKE_CURRENT_SOURCE_DIR}/core/include
        ${VTK_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${cv_bridge_INCLUDE_DIRS}
        ${cv_bridge_INCLUDE_DIRS}/cv_bridge
        ${rclcpp_LIBRARIES}
        ${visualization_msgs_LIBRARIES}
        ${geometric_shapes_LIBRARIES}
)

# Add dependencies and link libraries
ament_target_dependencies(ftv_ros
        rclcpp
        moveit_core
        moveit_ros_planning_interface
        xarm_msgs
        xarm_api
        xarm_description
        gazebo_ros
        gazebo_ros_pkgs
        cv_bridge
        sensor_msgs
        geometry_msgs
)

# Link libraries
target_link_libraries(ftv_ros
        viewpoint_core
        ${VTK_LIBRARIES}
        ${PCL_LIBRARIES}
)

# Ensure the configuration file is copied for the FTV executable
add_custom_command(TARGET ftv_ros POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/core/cfg/default.json
        $<TARGET_FILE_DIR:ftv_ros>/core/cfg/default.json
        COMMENT "Copying configuration.yaml for ftv_ros"
)

# Copy configuration file to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/core/cfg/default.json
        ${CMAKE_CURRENT_BINARY_DIR}/default.json
        COPYONLY)

# Copy the model files to the build directory
file(COPY ${CMAKE_SOURCE_DIR}/core/3d_models
        DESTINATION ${CMAKE_BINARY_DIR}/models)

# Install targets
install(TARGETS ftv_ros
        DESTINATION lib/${PROJECT_NAME})

# Install configuration file
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/core/cfg/default.json
        DESTINATION share/${PROJECT_NAME}/config)

# Install header files
install(DIRECTORY include/
        DESTINATION include/${PROJECT_NAME}
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# Install launch files
install(DIRECTORY
        launch
        DESTINATION share/${PROJECT_NAME}/
)

# Install the model files
install(FILES
        ${CMAKE_SOURCE_DIR}/models/object_model.sdf
        DESTINATION share/${PROJECT_NAME}/models
)

install(DIRECTORY
        ${CMAKE_SOURCE_DIR}/core/3d_models
        DESTINATION share/${PROJECT_NAME}/models
)

# Install URDF and Xacro files
install(DIRECTORY
        urdf
        DESTINATION share/${PROJECT_NAME}
)

# Install mesh files
install(DIRECTORY
        meshes
        DESTINATION share/${PROJECT_NAME}
)

# Install core resources
install(DIRECTORY
        ${CMAKE_SOURCE_DIR}/core/target_images
        DESTINATION share/${PROJECT_NAME}/core
)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies()
endif()

add_dependencies(ftv_ros viewpoint_core)
ament_package()

message(STATUS "Configuration complete for ${PROJECT_NAME} version ${PROJECT_VERSION}")
