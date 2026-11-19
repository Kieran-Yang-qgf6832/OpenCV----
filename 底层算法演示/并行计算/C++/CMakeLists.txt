cmake_minimum_required(VERSION 3.10)
project(parallel_computing CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 本演示要输出耗时/加速比，必须优化构建；缺省给 Release，否则 -O0 的数据没有意义。
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "构建类型" FORCE)
endif()

# 仅依赖 OpenCV 与线程库（std::mutex / OpenCV 并行后端）
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(parallel_computing parallel_computing.cpp)
target_include_directories(parallel_computing PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(parallel_computing PRIVATE ${OpenCV_LIBS} Threads::Threads)
