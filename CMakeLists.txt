cmake_minimum_required(VERSION 3.8)
project(randomized_kernel_clustering)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES main.c)
add_executable(randomized_kernel_clustering ${SOURCE_FILES})