project(CourseProject)

cmake_minimum_required(VERSION 3.11)

SET(CMAKE_PREFIX_PATH C:/eigen-3.4.0)
set(SOURCES ActivateFunction.cpp)
set(HEADERS ActivateFunction.h Matrix.h Layer.h Net.h)
add_library(lib STATIC ${SOURCES} ${HEADERS})

find_package(Eigen3 REQUIRED)

add_executable(CourseProject main.cpp ${SOURCES})

target_link_libraries(CourseProject lib Eigen3::Eigen   )
