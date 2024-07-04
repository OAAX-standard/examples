cmake_minimum_required(VERSION 3.10.2)

project(c_example)

# This compiler setting is required to use dlmopen, which is a GNU extension
add_definitions(-D_GNU_SOURCE)

# Include directories
add_executable(c_example src/main.c src/runtime_utils.c src/utils.c)

target_include_directories(c_example PUBLIC include)
target_link_libraries(c_example PUBLIC pthread dl jpeg)

# Copy files in the "artifacts" directory to the build directory
file(COPY artifacts DESTINATION ${CMAKE_BINARY_DIR})
file(COPY cpu-artifacts DESTINATION ${CMAKE_BINARY_DIR})
file(COPY hailo-artifacts DESTINATION ${CMAKE_BINARY_DIR})
