cmake_minimum_required(VERSION 2.8.12)
project(segmenter)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV QUIET)
if (OpenCV_FOUND)
  message("OpenCV found.")
else()
  message("OpenCV not found.")
endif()

include_directories(./include ${OpenCV_INCLUDE_DIRS})
add_executable(segmenter ./src/segmenter.cpp)
target_link_libraries(segmenter ${OpenCV_LIBS})

#TODO: compiler flags

