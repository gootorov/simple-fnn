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

Gradient Network::backpropagate(Eigen::VectorXd net_output, Eigen::VectorXd label) const {
    auto d_sigmoid = [](auto component) -> double {
        return component * (1.0 - component);
    };

    // initialize the gradient.
    auto gradient = Gradient(layers.size());

    // compute the error in the output layer.
    auto output_err = learning_rate * (net_output - label)
        .cwiseProduct(net_output.unaryExpr(d_sigmoid));

    // add that error as the gradient component.
    gradient(layers.size() - 1) = output_err;

    // add remaining errors to the gradient.
    for (int i = layers.size() - 2; i > -1; i--) {
        const auto& layer = layers[i];
        const auto& prev_layer = layers[i + 1];

        gradient(i) = learning_rate * layer.backpropagate(gradient(i + 1), prev_layer);
    }

    return gradient;
}

void Network::gradient_descent(Gradient gradient) {
    for (std::size_t i = layers.size() - 1; i > 1; i--) {
        auto& layer = layers[i];
        const auto& prev_layer = layers[i - 1];

        layer.gradient_descent(gradient(i), prev_layer);
    }
}

void Network::learn(const Data& training_data, const Labels& labels) {
    // TODO: Figure out a way to add element to a gradient
    // in which Eigen::Vector's are uninitialized?
    auto net_output = forward_propagate(training_data[0]);
    auto gradient = backpropagate(net_output, labels[0]);

    for (std::size_t i = 1; i < training_data.size(); i++) {
        const auto& image = training_data[i];
        const auto& label = labels[i];

        auto net_output = forward_propagate(image);
        gradient += backpropagate(net_output, label);
    }
    // TODO: What?
    for (std::size_t i = 0; i < layers.size(); i++) {
        gradient(i) /= training_data.size();
    }
    gradient_descent(gradient);
}

int Network::accuracy(const Data& data, const Labels& labels) {
    double correct{};
    for (std::size_t i = 0; i < data.size(); i++) {
        const auto& test_image = data[i];
        const auto& vectorized_label = labels[i];
        long int label{};
        vectorized_label.maxCoeff(&label);

        auto vectorized_prediction = forward_propagate(test_image);
        long int prediction{};
        vectorized_prediction.maxCoeff(&prediction);

        if (prediction == label) {
            correct++;
        }
    }
    return (correct / double(data.size())) * 100;
}

double Network::cost(const Data& training_data, const Labels& labels) {
    double cost{};

    for (std::size_t i = 0; i < training_data.size(); i++) {
        auto image = training_data[i];
        auto label = labels[i];

        auto prediction = forward_propagate(image);
        cost += (label - prediction).squaredNorm();
    }

    return cost / training_data.size();
}
