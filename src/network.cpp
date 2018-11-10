#include <Eigen/Core>
#include <cmath>

#include "network.hpp"
#include "layer.hpp"

Network::Network(std::size_t layers, std::size_t width) :layers{layers, Layer{width}} {
    // The biases of neurons in the first layer
    // should be initialized to 0 and its width must be 784.
    this->layers[0] = Layer{784, 0};
}

void Network::forward_propagate(const Eigen::VectorXd& input) {
    auto sigmoid = [](auto neuron) {
        return 1 / (1 + 1 / std::exp(neuron));
    };

    auto output = input;
    std::size_t is_first = 0;
    for (auto& layer : layers) {
        // Ugly hack to skip the first layer.
        if (is_first++ == 0) {
            continue;
        }

        output = this->weights * output + layer.get_neurons();
        output = output.unaryExpr(sigmoid);
    }
}
