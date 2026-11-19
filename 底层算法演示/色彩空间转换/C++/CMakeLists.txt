cmake_minimum_required(VERSION 3.10)
project(color_space_conversion CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 缺省给 Release：演示里的耗时/对比数据在以 -O0 构建时没有参考意义。
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "构建类型" FORCE)
endif()

# 仅依赖 OpenCV
find_package(OpenCV REQUIRED)

add_executable(color_space_conversion color_space_conversion.cpp)
target_include_directories(color_space_conversion PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(color_space_conversion PRIVATE ${OpenCV_LIBS})
