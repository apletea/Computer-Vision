cmake_minimum_required(VERSION 3.4)
project(final C CXX)
set(CMAKE_CXX_STANDARD 11)

set(Boost_INCLUDE_DIR /home/davinci/work/boost_1_61_0)
set(Boost_LIBRARY_DIR /home/davinci/work/boost_1_61_0/stage/lib)
set(Detector_INCLUDE_DIR 3rdparty/test_task/include)
set(CMAKE_BINARY_DIR lib)

FIND_FILE(Detector libtest_task.so 3rdparty/test_task/lib)
FIND_LIBRARY(Detector test_task 3rdparty/test_task/lib)


LINK_DIRECTORIES(${CMAKE_BINARY_DIR})
add_library(Detect SHARED IMPORTED )
set_target_properties(Detect PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/test_task.so )



find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(OpenCV REQUIRED)


add_executable(final main.cpp)

include_directories(${Boost_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Detector_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})
link_directories(${test_task})
link_directories($test_task_include)
target_link_libraries( final ${OpenCV_LIBS} )
target_link_libraries( final ${Boost_LIBRARIES} )
target_link_libraries( final ${Detector})
TARGET_LINK_LIBRARIES(final pthread)