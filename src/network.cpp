#include <iostream>

#include <Eigen/Core>

#include "internal.hpp"
#include "network.hpp"
#include "layer.hpp"

namespace NeuralNet {

Network::Network(std::size_t layers, std::size_t width, double learning_rate) :
    layers{layers, Layer{width, width}},
    learning_rate{learning_rate}
{
    this->layers.insert(this->layers.begin(), Layer{width, 784});

    this->layers.push_back(Layer{10, width});
}

Network::Network(std::size_t i_layer, int h_layers, std::size_t h_width,
        std::size_t o_layer,
        double learning_rate) :
    learning_rate{learning_rate}
{
    this->layers.push_back(Layer{h_width, i_layer});
    for (int i = 0; i < h_layers; i++) {
        this->layers.push_back(Layer{h_width, h_width});
    }
    this->layers.push_back(Layer{o_layer, h_width});
}

Network::Network(std::vector<std::size_t> layers, double learning_rate) :
    learning_rate{learning_rate}
{
    for (std::size_t i = 0; i < layers.size() - 1; i++) {
        const auto& layer = layers[i];
        const auto& next_layer = layers[i + 1];

        this->layers.push_back(Layer{next_layer, layer});
    }
}

Vec Network::forward_propagate(Vec input) {
    for (auto& layer : layers) {
        layer.forward_propagate(input);
    }

    return input;
}

Gradient Network::backpropagate(const Vec& net_output, const Vec& label) const {
    auto gradient = Gradient(layers.size());

    // compute the error in the output layer.
    const auto output_err = learning_rate * (net_output - label)
        .cwiseProduct(net_output.unaryExpr(&internal::d_sigmoid));

    // add that error as the gradient component.
    gradient(layers.size() - 1) = output_err;

    // add remaining errors to the gradient.
    for (int i = layers.size() - 2; i >= 0; i--) {
        const auto& layer = layers[i];
        const auto& prev_layer = layers[i + 1];

        gradient(i) = learning_rate * layer.backpropagate(gradient(i + 1), prev_layer);
    }

    return gradient;
}

void Network::gradient_descent(const Gradient& gradient) {
    for (int i = layers.size() - 1; i >= 0; i--) {
        auto& layer = layers[i];

        layer.gradient_descent(gradient(i));
    }
}

void Network::learn(const Data& training_data, const Labels& labels, bool debug) {
    for (std::size_t i = 0; i < training_data.size(); i++) {
        const auto& image = training_data[i];
        const auto& label = labels[i];

        const auto net_output = forward_propagate(image);
        const auto gradient = backpropagate(net_output, label);
        gradient_descent(gradient);

        if (debug) {
            std::cout << "Cost: " << (label - net_output).norm() << "\n";
        }
    }
}

int Network::accuracy(const Data& data, const Labels& labels) {
    using internal::argmax;

    int correct{};
    for (std::size_t i = 0; i < data.size(); i++) {
        const auto label = argmax(labels[i]);
        const auto prediction = argmax(forward_propagate(data[i]));
        if (prediction == label) {
            correct++;
        }
    }
    return (double(correct) / double(data.size())) * 100;
}

double Network::cost(const Data& data, const Labels& labels) {
    double cost{};

    for (std::size_t i = 0; i < data.size(); i++) {
        const auto& image = data[i];
        const auto& label = labels[i];

        const auto prediction = forward_propagate(image);
        cost += (label - prediction).squaredNorm();
    }

    return cost / data.size();
}

} // namespace NeuralNet
