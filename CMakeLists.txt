project(CourseProject)

cmake_minimum_required(VERSION 3.11)

set(SOURCES ActivateFunction.cpp Matrix.cpp)
set(HEADERS ActivateFunction.h Matrix.h)
add_library(lib STATIC ${SOURCES} ${HEADERS})

add_executable(CourseProject main.cpp ${SOURCES})

target_link_libraries(CourseProject lib)
