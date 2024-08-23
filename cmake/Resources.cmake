# cmake/Resources.cmake

# Custom target to copy necessary files to the output directory for both library and executable
add_custom_target(copy_resources ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PROJECT_SOURCE_DIR}/configuration.yaml
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/3d_models
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/3d_models
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/target_images
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/target_images
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PROJECT_SOURCE_DIR}/configuration.yaml
        $<TARGET_FILE_DIR:${PROJECT_NAME}_exe>
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/3d_models
        $<TARGET_FILE_DIR:${PROJECT_NAME}_exe>/3d_models
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${PROJECT_SOURCE_DIR}/target_images
        $<TARGET_FILE_DIR:${PROJECT_NAME}_exe>/target_images
        COMMENT "Copying configuration.yaml, 3d_models, and target_images directories to output directories for both library and executable"
)

# Add dependency to ensure resources are copied after both targets are built
if (TARGET ${PROJECT_NAME} AND TARGET ${PROJECT_NAME}_exe)
    add_dependencies(${PROJECT_NAME} copy_resources)
    add_dependencies(${PROJECT_NAME}_exe copy_resources)
else ()
    message(WARNING "Main targets ${PROJECT_NAME} and/or ${PROJECT_NAME}_exe not found. Include this file after creating the main targets.")
endif ()