cmake_minimum_required(VERSION 3.13)

project(c_example LANGUAGES C)

set(CMAKE_C_STANDARD 11)

set(JPEG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/jpeg/Windows)

add_executable(c_example
    src/main.c
    src/runtime.c
    src/tensors.c
    src/image.c)

target_include_directories(c_example PRIVATE include deps/oaax)

if(MSVC)
    target_compile_definitions(c_example PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(c_example PRIVATE -Wall -Wextra)
endif()

if(WIN32)
    # libjpeg is not installed on Windows: use the bundled one
    target_include_directories(c_example PRIVATE "${JPEG_DIR}/include")
    target_link_directories(c_example PRIVATE "${JPEG_DIR}/lib")
    target_link_libraries(c_example PRIVATE jpeg)
    file(COPY "${JPEG_DIR}/bin/jpeg62.dll" DESTINATION ${CMAKE_BINARY_DIR})
else()
    target_link_libraries(c_example PRIVATE jpeg ${CMAKE_DL_LIBS})
endif()

# Copy files in the "artifacts" directory to the build directory
file(COPY artifacts DESTINATION ${CMAKE_BINARY_DIR})
