#include <Eigen/Core>
#include <cmath>

#include "network.hpp"
#include "layer.hpp"

Network::Network(std::size_t layers, std::size_t width) :layers{layers, Layer{width, width}} {
    // The biases of neurons in the first layer
    // should be initialized to 0 and its width must be 28 x 28 = 784.
    this->layers.insert(this->layers.begin(), Layer{width, 784, 0});

    // The last layer is always has the width of 10.
    this->layers.push_back(Layer{10, width});
}

Eigen::VectorXd Network::forward_propagate(const Eigen::VectorXd input) const {
    // make a copy of the vector we're going to propagate.
    auto image = input;

    // propagate that vector.
    for (const auto& layer : layers) {
        layer.forward_propagate(image);
    }

    return image;
}
