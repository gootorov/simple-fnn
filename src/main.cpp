#include <iostream>

#include <Eigen/Core>

#include "mnist/mnist_reader.hpp"
#include "mnist/mnist_utils.hpp"

double normalize(double pixel) {
    return pixel / 255.0;
}

int main() {
    // load mnist data
    auto mnist = mnist::read_dataset<std::vector, std::vector, double, uint8_t>(MNIST_DATA_LOCATION);

    // convert std::vector's to Eigen::vector's.
    std::vector<Eigen::VectorXd> training_images{};
    std::vector<Eigen::VectorXd> test_images{};
    for (auto& image : mnist.training_images) {
        training_images.push_back(Eigen::VectorXd::Map(image.data(), image.size()));
    }
    for (auto& image : mnist.test_images) {
        test_images.push_back(Eigen::VectorXd::Map(image.data(), image.size()));
    }

    //mnist::binarize_dataset(mnist);
    return 0;
}
