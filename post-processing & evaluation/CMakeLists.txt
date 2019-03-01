cmake_minimum_required(VERSION 3.8)
project(Re_X)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES
	main.cpp
	MyLine.cpp
	MyLine.h
	MyPoint.cpp
	MyPoint.h)
add_executable(Re_X ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(Re_X ${OpenCV_LIBS})
