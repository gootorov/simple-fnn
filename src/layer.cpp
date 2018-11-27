#include <random>
#include "layer.hpp"

double Layer::random() const {
    std::random_device seed{};
    std::mt19937 engine{seed()};
    std::uniform_real_distribution<double> random_value{-1.0, 1.0};

    return random_value(engine);
}

Layer::Layer(std::size_t width, std::size_t width_prev_layer) :
    weights{width, width_prev_layer},
    neurons{width},
    activation{width}
{
    auto rand = [this](double) -> double {
        return this->random();
    };

    // Initialize weights and biases to random values.
    this->neurons = neurons.unaryExpr(rand);
    this->weights = weights.unaryExpr(rand);
}

Layer::Layer(std::size_t width, std::size_t width_prev_layer, double bias) :
    weights{width, width_prev_layer},
    neurons{width},
    activation{width}
{
    this->neurons.fill(bias);

    // Initialize weights and to random values.
    auto rand = [this](double) -> double {
        return this->random();
    };
    this->weights = weights.unaryExpr(rand);
}

void Layer::forward_propagate(Eigen::VectorXd& input) {
    auto sigmoid = [](auto component) -> double {
        return 1.0 / (1.0 + 1.0 / std::exp(component));
    };

    // propagate the vector
    this->activation = (weights * input + neurons).unaryExpr(sigmoid);
    input = activation;
}

Eigen::VectorXd Layer::backpropagate(Eigen::VectorXd prev_err, const Layer& prev_layer) const {
    const auto d_sigmoid = [](auto component) -> double {
        return component * (1.0 - component);
    };

    return (prev_layer.weights.transpose() * prev_err)
        .cwiseProduct(prev_layer.activation.unaryExpr(d_sigmoid));
}

void Layer::gradient_descent(Eigen::VectorXd gradient, const Layer& prev_layer) {
    this->weights -= gradient * prev_layer.activation.transpose();
    this->neurons -= gradient;
}
