cmake_minimum_required(VERSION 3.7)
project(reOpenSURF)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

set(SOURCE_FILES main.cpp)
add_executable(reOpenSURF ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(reOpenSURF ${OpenCV_LIBS})
target_link_libraries(reOpenSURF ${dlib_LIBS})