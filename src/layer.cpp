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

const Eigen::VectorXd& Layer::get_neurons() const {
    return this->neurons;
}

Eigen::VectorXd& Layer::get_neurons_mut() {
    return this->neurons;
}
