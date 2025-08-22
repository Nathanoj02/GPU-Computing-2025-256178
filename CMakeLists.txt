cmake_minimum_required(VERSION 3.10)

project(GPUComputing)

# Find OpenCV4 specifically
find_package(OpenCV 4.2 REQUIRED)

# Debug: Print OpenCV information
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(inc)

# Define source files variable
set(SOURCES
    src/main.cpp
    src/cluster.c
)

add_executable(GPUComputing ${SOURCES})

target_link_libraries(GPUComputing ${OpenCV_LIBS} m)