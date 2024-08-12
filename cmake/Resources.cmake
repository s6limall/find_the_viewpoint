# cmake/Resources.cmake

# Custom target to copy necessary files to the output directory
add_custom_target(copy_resources ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PROJECT_SOURCE_DIR}/configuration.yaml
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/3d_models
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/3d_models
        COMMENT "Copying configuration.yaml and 3d_models directory to output directory"
)

# Add dependency to ensure resources are copied after the main target is built
# This will work as long as this file is included after the main target is created
if (TARGET ${PROJECT_NAME})
    add_dependencies(${PROJECT_NAME} copy_resources)
else ()
    message(WARNING "Main target ${PROJECT_NAME} not found. Include this file after creating the main target.")
endif ()
