cmake_minimum_required(VERSION 3.12.2)
project(neural_net)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -Werror")

# Debug/Release builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

# neural_net
include_directories(include)
file(GLOB SOURCE_FILES src/*.cpp)
file(GLOB HEADER_FILES include/*.h)

# eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE PATHS lib/eigen/eigen-3.3.5/build)

# mnist
find_package(MNIST PATHS lib/mnist)
#include_directories(lib/mnist/include)
include_directories(${MNIST_INCLUDE_DIR})

add_executable(neural_net ${SOURCE_FILES} ${HEADER_FILES})

# link libraries
target_link_libraries(neural_net Eigen3::Eigen)

# Pass MNIST data to the neural_net
target_compile_definitions(neural_net PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
