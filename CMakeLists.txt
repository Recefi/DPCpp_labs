cmake_minimum_required(VERSION 3.23.5)

# Modifying the compiler should be done before the project line
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda,spir64_x86_64")

project(DPCpp_labs LANGUAGES CXX)  # tells CMake the name of this project and that it uses C++

add_executable(lab1 lab1.cpp)
