cmake_minimum_required(VERSION 3.13)

project(yolox_example LANGUAGES C)

set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The YOLOX pre/post-processing: no OAAX, no I/O, so the tests link it alone
add_library(yolox STATIC src/yolox.c)
target_include_directories(yolox PUBLIC include)

add_executable(yolox_host
    src/main.c
    src/runtime.c
    src/tensors.c
    src/image.c)
target_link_libraries(yolox_host PRIVATE yolox)
target_include_directories(yolox_host PRIVATE deps/oaax)
# The single-header libraries are third-party code: no warnings from them
target_include_directories(yolox_host SYSTEM PRIVATE third_party)

if(MSVC)
  target_compile_definitions(yolox_host PRIVATE _CRT_SECURE_NO_WARNINGS)
  target_compile_definitions(yolox PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(yolox_host PRIVATE -Wall -Wextra)
  target_compile_options(yolox PRIVATE -Wall -Wextra)
  target_link_libraries(yolox PUBLIC m)
  target_link_libraries(yolox_host PRIVATE ${CMAKE_DL_LIBS})
endif()

# Unit tests of the YOLOX code: `ctest` in the build directory
enable_testing()
add_executable(test_yolox tests/test_yolox.c)
target_link_libraries(test_yolox PRIVATE yolox)
add_test(NAME yolox COMMAND test_yolox)
