#include <random>
#include "layer.hpp"

double Layer::random() const {
    std::random_device seed{};
    std::mt19937 engine{seed()};
    std::uniform_real_distribution<double> random_value{-1.0, 1.0};
}

Layer::Layer(std::size_t width) {
    this->neurons = Eigen::VectorXd{width};
    this->neurons.fill(random());

    this->weights = Eigen::MatrixXd{width, width};
    this->weights.fill(random());
}

Layer::Layer(std::size_t width, double bias) {
    this->neurons = Eigen::VectorXd{width};
    this->neurons.fill(bias);

    this->weights = Eigen::MatrixXd{width, width};
    this->weights.fill(random());
}

const Eigen::VectorXd& Layer::get_neurons() const {
    return this->neurons;
}

Eigen::VectorXd& Layer::get_neurons_mut() {
    return this->neurons;
}
