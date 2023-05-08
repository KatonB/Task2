cmake_minimum_required(VERSION 3.23)
project(Task2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")

# Поиск библиотеки MPI
find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

add_executable(Task2 main.cpp)

# Связывание с библиотекой MPI
target_link_libraries(Task2 ${MPI_CXX_LIBRARIES})



