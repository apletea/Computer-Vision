cmake_minimum_required(VERSION 3.5)
project(test_task)

set(Boost_INCLUDE_DIR /usr/local/src/boost_1_46_1)
set(Boost_LIBRARY_DIR /usr/local/src/boost_1_46_1/stage/lib)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)


set(CMAKE_CXX_STANDARD 11)

add_library(test_task SHARED face_detector.cpp face_detector.h ThreadPool.cpp ThreadPool.h)

include_directories(${Boost_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIR})
target_link_libraries( test_task ${OpenCV_LIBS} )
target_link_libraries( test_task ${Boost_LIBRARIES} )