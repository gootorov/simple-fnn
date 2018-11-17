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
    neurons{width}
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
    neurons{width}
{
    this->neurons.fill(bias);

    // Initialize weights and to random values.
    auto rand = [this](double) -> double {
        return this->random();
    };
    this->weights = weights.unaryExpr(rand);
}

void Layer::forward_propagate(Eigen::VectorXd& input) const {
    auto sigmoid = [](auto component) -> double {
        return 1.0 / (1.0 + 1.0 / std::exp(component));
    };

    // propagate the vector
    input = (weights * input + neurons).unaryExpr(sigmoid);
}
