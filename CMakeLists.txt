cmake_minimum_required(VERSION 3.3)
project(main)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14")

set(SOURCE_FILES main.cpp mnist_loader.cpp network.cpp)

include_directories(.)

add_executable(main ${SOURCE_FILES})

LINK_DIRECTORIES(/usr/lib/x86_64-linux-gnu/ /usr/include/libpng12)
TARGET_LINK_LIBRARIES(main png12)

