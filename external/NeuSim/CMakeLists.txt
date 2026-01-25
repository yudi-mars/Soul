cmake_minimum_required(VERSION 3.25.0 FATAL_ERROR)

project(stdexecExample)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

set(NC_SIM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE STRING "")

add_subdirectory(src)

add_executable(main example.cpp)
target_link_libraries(main neu_sim)
