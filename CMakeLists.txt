PROJECT(CourseProject)

CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

SET(CMAKE_CXX_STANDARD 17)

SET(CMAKE_PREFIX_PATH C:/eigen-3.4.0)

FIND_PACKAGE(Eigen3 REQUIRED)

SET(SOURCES ActivateFunction.cpp test.cpp)
SET(HEADERS ActivateFunction.h Layer.h Net.h test.h)
ADD_LIBRARY(lib STATIC ${SOURCES} ${HEADERS})
TARGET_LINK_LIBRARIES(lib Eigen3::Eigen)
ADD_LIBRARY(test_lib STATIC test.cpp test.h)
TARGET_LINK_LIBRARIES(test_lib Eigen3::Eigen)

ADD_EXECUTABLE(CourseProject main.cpp ${SOURCES})

TARGET_LINK_LIBRARIES(CourseProject lib test_lib)
