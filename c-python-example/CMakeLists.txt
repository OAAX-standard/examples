cmake_minimum_required(VERSION 3.10.2)

project(c_python_example)

# Cross-compilation settings
set(CROSS_NAME "X86_64")
set(CROSS_ROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0")
set(COMPILER_PREFIX "x86_64-unknown-linux-gnu-")
set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
set(CMAKE_LINKER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)

# Include directories
file(GLOB_RECURSE SRC "src/*.c")
message("SRC: ${SRC}")
add_executable(c_python_example ${SRC})

target_include_directories(c_python_example PUBLIC include)
target_link_libraries(c_python_example PUBLIC pthread dl)