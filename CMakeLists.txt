cmake_minimum_required(VERSION 3.12.2)
project(neural_net)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -Werror")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)

file(GLOB SOURCE_FILES
    "*.h"
    "*.cpp"
)

include_directories(../include)
add_executable(target/neural_net ${SOURCE_FILES})
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
