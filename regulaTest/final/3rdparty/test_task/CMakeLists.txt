cmake_minimum_required(VERSION 3.5)
project(test_task C CXX)

set(Boost_INCLUDE_DIR /home/davinci/work/boost_1_61_0)
set(Boost_LIBRARY_DIR /home/davinci/work/boost_1_61_0/stage/lib)


find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_REQUIRED_FLAGS -m32)
add_library(test_task SHARED src/face_detector.cpp include/face_detector.h src/ThreadPool.cpp include/ThreadPool.h)
#set_target_properties(test_task PROPERTIES COMPILE_FLAGS "-m32" LINK_FLAGS "-m32")



link_directories(include)
link_directories(src)

include_directories(${Boost_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src)
include_directories(include)
link_directories(${Boost_LIBRARY_DIR})
target_link_libraries( test_task ${OpenCV_LIBS} )
target_link_libraries( test_task ${Boost_LIBRARIES} )
