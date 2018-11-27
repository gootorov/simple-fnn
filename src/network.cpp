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

Eigen::VectorXd Network::forward_propagate(Eigen::VectorXd image) {
    // propagate that vector.
    for (auto& layer : layers) {
        layer.forward_propagate(image);
    }

    return image;
}

Network::Array Network::backpropagate(Eigen::VectorXd net_output, Eigen::VectorXd label) const {
    auto d_sigmoid = [](auto component) -> double {
        return component * (1.0 - component);
    };

    // initialize the gradient.
    auto gradient = Array(layers.size());

    // compute the error in the output layer.
    auto output_err = learning_rate * (net_output - label)
        .cwiseProduct(net_output.unaryExpr(d_sigmoid));

    // add that error as the gradient component.
    gradient(layers.size() - 1) = output_err;

    // add remaining errors to the gradient.
    for (std::size_t i = layers.size() - 2; i > 0; i--) {
        const auto& layer = layers[i];
        const auto& prev_layer = layers[i + 1];

        gradient(i) = learning_rate * layer.backpropagate(gradient(i + 1), prev_layer);
    }

    return gradient;
}

double Network::cost(const TrainingData& training_data, const Labels& labels) {
    double cost{};

    for (std::size_t i = 0; i < training_data.size(); i++) {
        auto image = training_data[i];
        auto label = labels[i];

        auto prediction = forward_propagate(image);
        cost += (label - prediction).squaredNorm();
    }

    return cost / training_data.size();
}
