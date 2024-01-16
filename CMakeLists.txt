PROJECT(CourseProject)

CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

SET(CMAKE_CXX_STANDARD 17)

SET(CMAKE_CXX_STANDARD_REQUIRED ON)

INCLUDE_DIRECTORIES(Eigen EigenRand)

SET(SOURCES ActivationFunction.cpp Layer.cpp Net.cpp test.cpp exception.cpp)
ADD_LIBRARY(lib STATIC ${SOURCES})
ADD_LIBRARY(test_lib STATIC test.cpp)

ADD_EXECUTABLE(CourseProject main.cpp ${SOURCES})

TARGET_LINK_LIBRARIES(CourseProject lib test_lib)
