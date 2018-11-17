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
    this->neurons.fill(random());

    this->weights.fill(random());
}

Layer::Layer(std::size_t width, std::size_t width_prev_layer, double bias) :
    weights{width, width_prev_layer},
    neurons{width}
{
    this->neurons.fill(bias);

    this->weights.fill(random());
}

void Layer::forward_propagate(Eigen::VectorXd& input) const {
    auto sigmoid = [](auto component) {
        return 1 / (1 + 1 / std::exp(component));
    };

    // propagate the vector
    input = weights * input + neurons;
    // apply the sigmoid fn to each component.
    input = input.unaryExpr(sigmoid);
}
