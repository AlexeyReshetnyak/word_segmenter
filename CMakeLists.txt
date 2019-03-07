cmake_minimum_required(VERSION 2.8.12)
project(word_segmenter)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV QUIET)
if (OpenCV_FOUND)
  message("OpenCV found.")
else()
  message("OpenCV not found.")
endif()

add_executable(word_segmenter ./src/word_segmenter.cpp)
target_link_libraries(word_segmenter ${OpenCV_LIBS})

#TODO: compiler flags

