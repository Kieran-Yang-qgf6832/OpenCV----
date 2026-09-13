cmake_minimum_required(VERSION 3.10)
project(background_subtraction CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 仅依赖 OpenCV
find_package(OpenCV REQUIRED)

add_executable(background_subtraction background_subtraction.cpp)
target_include_directories(background_subtraction PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(background_subtraction PRIVATE ${OpenCV_LIBS})
