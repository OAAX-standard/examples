cmake_minimum_required(VERSION 3.10)
project(ultralytics-inference)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add CLI11
add_subdirectory(external/CLI11)

# Add spdlog for logging
add_subdirectory(external/spdlog)

# Add OAAX tools
add_subdirectory(external/oaax_tools/c-utilities)

# Add nlohmann/json for JSON handling
add_subdirectory(external/json)

# Add OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/single_headers
)

# Add source files
add_executable(ultralytics_inference main.cpp input_handler.cpp preprocess.cpp postprocess.cpp visualize.cpp cli.cpp config.cpp logger.cpp threads.cpp runtime.cpp tensors.cpp)

# Link CLI11 to your executable
target_link_libraries(ultralytics_inference PRIVATE CLI11::CLI11)
target_link_libraries(ultralytics_inference PRIVATE spdlog::spdlog)
target_link_libraries(ultralytics_inference PRIVATE c_utilities)
target_link_libraries(ultralytics_inference PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(ultralytics_inference PRIVATE ${OpenCV_LIBS})

# Enable debugging and sanitizers for memory issues
# Note: Leak sanitizer removed due to ONNX Runtime compatibility issues
# target_link_libraries(yolov8_inference PRIVATE -fsanitize=address)