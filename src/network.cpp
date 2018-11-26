#include <Eigen/Core>
#include <cmath>

#include "network.hpp"
#include "layer.hpp"

Network::Network(std::size_t layers, std::size_t width, double learning_rate) :
    layers{layers, Layer{width, width}},
    learning_rate{learning_rate}
{
    // The biases of neurons in the first layer
    // should be initialized to 0 and its width must be 28 x 28 = 784.
    this->layers.insert(this->layers.begin(), Layer{width, 784});

    // The last layer always has the width of 10.
    this->layers.push_back(Layer{10, width});
}

Eigen::VectorXd Network::forward_propagate(Eigen::VectorXd image) const {
    // propagate that vector.
    for (const auto& layer : layers) {
        layer.forward_propagate(image);
    }

    return image;
}

double Network::cost(const TrainingData& training_data, const Labels& labels) const {
    double cost{};

    for (std::size_t i = 0; i < training_data.size(); i++) {
        auto image = training_data[i];
        auto label = labels[i];

        auto prediction = forward_propagate(image);
        cost += (label - prediction).squaredNorm();
    }

    return cost / training_data.size();
}
